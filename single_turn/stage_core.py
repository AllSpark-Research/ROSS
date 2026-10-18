#!/usr/bin/env python3
"""Core proposal, span resolution, coherence audit, and revision utilities."""
from __future__ import annotations
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "seg_v1", str(Path(__file__).with_name("span_utils.py")))
seg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(seg)

_spec2 = importlib.util.spec_from_file_location(
    "prefilter", str(Path(__file__).with_name("prefilter.py")))
pref = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(pref)

API, KEY, MODEL = seg.API, seg.KEY, seg.MODEL
ROLLOUT, OUTDIR, SELECTED = seg.ROLLOUT, seg.OUTDIR, seg.SELECTED
_find, parse_json, build_user = seg._find, seg.parse_json, seg.build_user

# ---------- reused post-hoc heuristics ----------
STRONG_ERROR_RE = pref.STRONG_ERROR_RE  # from prefilter.py
ANSWER_RE = pref.ANSWER_RE
# extra triggers not in STRONG_ERROR_RE that we've seen mismark whole_clean
EXTRA_TRIG_RE = re.compile(
    r"\b(wait|hmm|actually,?\s+(?:no|wait)|"
    r"let'?s\s+(?:re-?evaluate|re-?consider|re-?calculate|re-?check|redo|try\s+another|start\s+over)|"
    r"impossible|scratch\s+that)\b", re.IGNORECASE)


def whole_clean_suspect(response: str) -> dict:
    """C3: quick heuristic scan to catch cases like math/551c2083#0."""
    hits = {}
    if STRONG_ERROR_RE.search(response):
        hits["strong_error_phrase"] = STRONG_ERROR_RE.search(response).group(0)
    ans = list(ANSWER_RE.finditer(response))
    if len(ans) >= 2:
        hits["duplicated_answer"] = len(ans)
    et = EXTRA_TRIG_RE.findall(response)
    if len(et) >= 3:
        hits["hedging_triggers"] = len(et)
    return hits


# ---------------------------------------------------------------- pass-1 prompt
_V1_SYS = seg.SYSTEM_PROMPT

# Rebuild MASK guidance (P3) — clarify: mask only WRONG content, not verbose-correct.
_MASK_ORIG = """MASK (do NOT put in keep-spans):
  - wrong assumptions, wrong derivations, arithmetic errors, dead ends;
  - abandoned attempts / discarded candidate solutions;
  - the self-correction trigger and hedging ("wait, this is wrong", "let me redo",
    "actually my previous...", "hmm, let me reconsider");
  - speculation that does not end up in the final solution;
  - for code: discarded code versions, hard-coded shortcuts, wrong algorithm talk."""

_MASK_NEW = """MASK (do NOT put in keep-spans) — CORRECTNESS-BASED criteria:
  - wrong assumptions, wrong derivations, arithmetic errors, dead ends;
  - abandoned attempts / discarded candidate solutions that were declared wrong or
    impossible by the model itself;
  - the self-correction trigger text tightly bound to a wrong branch ("wait, this
    is wrong", "let me redo", "actually my previous...", "hmm, that gives a
    contradiction, let me reconsider");
  - speculation whose conclusion is NEVER used in the final solution;
  - for code: discarded code versions, hard-coded shortcuts, wrong algorithm talk.

DO NOT MASK just because content is redundant or repetitive:
  - a correct second verification of the same answer is safe to keep (it teaches
    self-checking);
  - a "Let's double-check" block that computes the SAME correct value again is
    keep-worthy unless the second computation reaches a different conclusion.
  - the discriminator is CORRECTNESS OF THE INTERMEDIATE CLAIM, not tone."""

# Anti-whole_clean hard rule (P2)
_ANTI_WC = r"""

WHOLE_CLEAN HARD RULE (STRICT)
Setting whole_clean=true is a strong claim: the entire response is worth imitating
verbatim. Set whole_clean=true ONLY when the response is a single monotone-forward
derivation. If ANY of the following signals appear anywhere in the response, then
whole_clean MUST be false and you MUST either produce keep_spans OR set
has_clean_final=false:

  - explicit correction triggers: "Wait", "Actually", "Hmm", "Let me reconsider",
    "Let's re-evaluate", "Let's re-calculate", "scratch that"
  - the word "Impossible" applied to an attempted ansatz the model itself tried
    (e.g. "Try a_m = C·m ... Impossible; Try a_m = m^2 ... Impossible")
  - two or more distinct candidate forms/approaches tried and rejected
  - a numeric/formal claim later contradicted by the model (e.g. concludes
    "k ≤ 2015" then later concludes "k ≤ 2016")
  - duplicated final Answer lines (e.g. "Answer: 2016\nAnswer: 2016")

If you're unsure whether there's a wrong branch after seeing any of these signals,
default to a span annotation, NOT whole_clean.

"""

# Insert MASK rewrite + anti-WC rule + then keep v2's SELF-CONTAINMENT block
_V3_SYS = _V1_SYS.replace(_MASK_ORIG, _MASK_NEW)
_V3_SYS = _V3_SYS.replace(
    "IMPORTANT\n",
    _ANTI_WC.lstrip() + "\n" +
    r"""SELF-CONTAINMENT (CRITICAL)
The kept spans, read in order with masked parts removed, MUST form a solution that
stands on its own. Every variable, case split, enumeration, definition or intermediate
result that a kept span uses MUST be introduced/justified inside some kept span.
- Do NOT mask a wrong branch if that removes setup/definitions a later kept span relies
  on. Extend the keep span to include the minimal correct bridging text instead.
- If the needed setup exists ONLY entangled with the mistake and cannot be cleanly
  kept, set has_clean_final=false.
- Never produce a kept solution with dangling references or unexplained jumps.

IMPORTANT
""", 1)

# Extra Example 5 (self-containment) — reuse v2's wording briefly
_EXAMPLE_5 = r"""

Example 5 - SELF-CONTAINMENT (avoid dangling references):
A response defines "n=2000+10c+d so S(n)=2+c+d" ONLY inside a wrong branch, and the
corrected part abruptly starts "If c=1, ...". Keeping only the "If c=1 ..." span is
BAD (c undefined in kept text). Instead extend the keep span to include a correct
restatement of the setup that introduces c.
```json
{"whole_clean": false, "has_clean_final": true,
 "keep_spans": [{"start_anchor": "For a=2, n=2000+10c+d, so S(n)=2+c+d", "end_anchor": "so n = 2011 is a solution.", "why": "self-contained corrected case-1 derivation incl. its setup"}],
 "mask_spans_summary": ["wrong-RHS branch and 'Wait/Why/Ah' debugging"],
 "reason": "kept a self-contained corrected derivation, not a dangling fragment",
 "risk_tags": ["self_correction","wrong_branch"]}
```
"""

# NEW Example 6 (P1) — long mostly-correct with a short mid-response error branch
_EXAMPLE_6 = r"""

Example 6 - LONG mostly-correct response with SHORT mid-response error branch
(this is the pattern most likely to be mis-classified as whole_clean):

Response derives an inequality k ≤ 2016 correctly for ~10 paragraphs, then briefly
concludes "So we need 2014 ≥ k-1 → k ≤ 2015" (wrong: the model confused the max
|m-n| bound), then writes "Wait, max difference is k-1 (between 1 and k). So k ≤ 2015"
still wrong direction, then re-examines the constraint domain m+n ≤ k, and after a
few paragraphs re-derives k ≤ 2016 and confirms constructibility, ending with the
correct Answer: 2016. There are ALSO two abandoned ansätze earlier ("Try a_m = Cm ...
Impossible", "Try a_m = m^2 ... Impossible"). The final answer is correct.

whole_clean=false is REQUIRED (multiple wrong sub-branches). Correct annotation
keeps the correct setup, MASKS the "Try a_m = Cm/m^2 → Impossible" attempts, MASKS
the "Wait ... k ≤ 2015" wrong-conclusion segment, KEEPS the correct re-derivation
and the final answer.
```json
{"whole_clean": false, "has_clean_final": true,
 "keep_spans": [
   {"start_anchor": "To determine the largest possible value $k$", "end_anchor": "So the largest possible value is 2016.", "why": "correct setup + correct re-derivation + confirmation, minus the two rejected ansätze and the k≤2015 wrong branch"}
 ],
 "mask_spans_summary": ["'Try a_m = Cm ... Impossible' abandoned ansatz","'Try a_m = m^2 ... Impossible' abandoned ansatz","'Wait ... k ≤ 2015' wrong-conclusion segment and its follow-up"],
 "reason": "long response with two abandoned ansätze and one wrong-conclusion segment; final answer correct",
 "risk_tags": ["abandoned_attempt","wrong_branch","self_correction"]}
```
Note: even a SHORT wrong-conclusion segment inside a long correct response requires
whole_clean=false. Do NOT collapse to whole_clean just because the response is
"mostly" correct — that would teach the model the wrong sub-branch too.
"""

SYSTEM_PROMPT_V3 = _V3_SYS + _EXAMPLE_5 + _EXAMPLE_6


# ---------------------------------------------------------------- pass-2 prompt
# P5: audit also scans for MISSED wrong branches, not just kept-text coherence.
COHERENCE_SYSTEM_V3 = r"""You audit a SEGMENT-LEVEL annotation of a model response. Two checks:

CHECK A - kept-text self-containment.
The masked regions are shown as "[... masked ...]" between kept spans. The kept
text alone must be coherent and reach the correct final answer.
Flag a problem if the kept text:
- uses a variable / case / enumeration that was never introduced in the kept text
  (dangling reference caused by masking its setup);
- has an unexplained logical jump where a masked region clearly held a needed step;
- does not reach the correct final answer, or the final answer line is missing.

CHECK B - missed wrong branches in the full response.
Scan the FULL response (provided below the kept text) and list any wrong / abandoned
/ speculative content that was NOT covered by a mask region and NOT inside a kept
span that is itself correct. Signals to look for:
  - "Wait ... [some wrong conclusion]" segments not marked masked
  - "Try X ... Impossible" / "Try Y ... Impossible" abandoned ansätze not marked
    masked
  - two conflicting numeric conclusions where the wrong one is inside a kept span
  - duplicated final Answer lines both inside kept text

Return JSON only in a fenced block:
```json
{"coherent": true/false,
 "reaches_correct_answer": true/false,
 "severity": "none" | "minor" | "major",
 "issues": ["short concrete issue A"],
 "missed_masks": ["short quoted wrong content from the full response that should have been masked but wasn't"]}
```
severity=major only for real dangling refs / broken logic / wrong-or-missing answer
OR when missed_masks is non-empty. severity=minor for terse-but-followable reasoning
with no missed masks. severity=none if fully self-contained AND missed_masks empty.
"""


def call(system: str, user: str, max_tokens: int = 32768, temperature: float = 0.0) -> str:
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "stream": False, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode(),
                                 headers=seg.api_headers(),
                                 method="POST")
    body = json.loads(urllib.request.urlopen(req, timeout=300).read())
    msg = body["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def parse_obj(text: str, need):
    blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    cands = list(reversed(blocks)) + list(reversed(re.findall(r"\{.*\}", text, re.DOTALL)))
    for c in cands:
        try:
            o = json.loads(c)
        except Exception:
            continue
        if isinstance(o, dict) and any(k in o for k in need):
            return o
    return None


def resolve_spans(response: str, parsed: dict) -> dict:
    """Same as v2's resolve_spans (kept intact)."""
    L = len(response)
    if parsed.get("whole_clean"):
        return {"mode": "whole_clean", "kept_spans": [[0, L]], "kept_frac": 1.0, "spans_status": []}
    ks = parsed.get("keep_spans") or []
    if not parsed.get("has_clean_final") or not ks:
        return {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0, "spans_status": []}

    starts = [_find(response, sp.get("start_anchor", "")) for sp in ks]
    ends = [_find(response, sp.get("end_anchor", "")) for sp in ks]
    spans, status = [], []
    for i in range(len(ks)):
        s = starts[i]
        if not s:
            status.append("start_missing"); continue
        e = ends[i]
        if e and e[1] > s[0]:
            spans.append([s[0], e[1]]); status.append("ok"); continue
        nxt = next((starts[j][0] for j in range(i + 1, len(ks))
                    if starts[j] and starts[j][0] > s[0]), None)
        if nxt is not None:
            spans.append([s[0], nxt]); status.append("end_missing->next_start")
        else:
            spans.append([s[0], L]); status.append("end_missing->to_end")

    if not spans:
        s = _find(response, ks[-1].get("start_anchor", ""))
        if s:
            return {"mode": "fallback_last_to_end", "kept_spans": [[s[0], L]],
                    "kept_frac": (L - s[0]) / L, "spans_status": status}
        return {"mode": "fallback_tail_50pct", "kept_spans": [[L // 2, L]],
                "kept_frac": 0.5, "spans_status": status}

    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1] or response[merged[-1][1]:a].strip() == "":
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    frac = sum(b - a for a, b in merged) / L
    mode = "spans_ok" if all(x == "ok" for x in status) else "spans_partial"
    return {"mode": mode, "kept_spans": merged, "kept_frac": frac, "spans_status": status}


def reconstruct(response: str, kept_spans) -> str:
    parts, prev = [], 0
    for a, b in kept_spans:
        if a > prev and response[prev:a].strip():
            parts.append("[... masked ...]")
        parts.append(response[a:b]); prev = b
    if prev < len(response) and response[prev:].strip():
        parts.append("[... masked ...]")
    return "\n".join(parts)


def coherence_audit(sample, kept_text: str) -> dict:
    dom = sample["metadata"]["domain"]
    gt = ("correct_answer = " + str((sample["metadata"].get("reward_spec") or {}).get("ground_truth"))
          if dom == "math" else "target_output = " + str(sample["reward"].get("target_output")))
    user = (f"[DOMAIN] {dom}\n[GROUND_TRUTH] {gt}\n\n[PROBLEM]\n{sample['prompt']}\n\n"
            f"[KEPT SOLUTION TO AUDIT]\n{kept_text}\n\n"
            f"[FULL ORIGINAL RESPONSE — for CHECK B]\n{sample['response']}\n\nAudit now.")
    raw = call(COHERENCE_SYSTEM_V3, user)
    parsed = parse_obj(raw, ("severity", "coherent", "missed_masks")) or {}
    return {"coherent": parsed.get("coherent"),
            "reaches_correct_answer": parsed.get("reaches_correct_answer"),
            "severity": parsed.get("severity", "unknown"),
            "issues": parsed.get("issues", []),
            "missed_masks": parsed.get("missed_masks", []),
            "raw": raw}


# P4: stronger revise prompt + higher temperature to break the model out of repeating itself
def revise(sample, prev_parsed, audit) -> str:
    user = (build_user(sample) +
            "\n\n[PREVIOUS ANNOTATION — INSUFFICIENT]\n" + json.dumps(prev_parsed, ensure_ascii=False) +
            "\n\n[COHERENCE AUDIT ISSUES]\n" + json.dumps(audit.get("issues", []), ensure_ascii=False) +
            "\n\n[MISSED WRONG BRANCHES the previous annotation FAILED to mask]\n" +
            json.dumps(audit.get("missed_masks", []), ensure_ascii=False) +
            "\n\nYour PREVIOUS annotation was insufficient. You MUST produce a NEW annotation "
            "that DIFFERS from the previous one. Do NOT return identical keep_spans.\n"
            "Choose exactly ONE of these three actions:\n"
            "  (a) EXTEND — enlarge existing keep span(s) to include the minimal correct "
            "bridging setup so the kept text is self-contained;\n"
            "  (b) SHRINK — remove or narrow keep span(s) that reference undefined variables "
            "or that contain missed wrong content, and/or add explicit new mask spans covering "
            "the missed wrong branches;\n"
            "  (c) DROP — if no clean self-contained correct solution can be extracted, set "
            "has_clean_final=false and keep_spans=[].\n"
            "Return the json block only.")
    return call(SYSTEM_PROMPT_V3, user, temperature=0.5)


def _same_spans(a_parsed, b_parsed) -> bool:
    """C2: are two annotations effectively identical?"""
    if a_parsed is None or b_parsed is None:
        return False
    ka = a_parsed.get("keep_spans") or []
    kb = b_parsed.get("keep_spans") or []
    if len(ka) != len(kb):
        return False
    for x, y in zip(ka, kb):
        if x.get("start_anchor") != y.get("start_anchor") or x.get("end_anchor") != y.get("end_anchor"):
            return False
    return a_parsed.get("whole_clean") == b_parsed.get("whole_clean") \
        and a_parsed.get("has_clean_final") == b_parsed.get("has_clean_final")


def annotate(job):
    dom, ph, ri, sample = job
    r = sample["reward"]
    passed = (r.get("follow_all_instructions") if dom == "if" else r.get("score") == 1.0)
    rec = {"domain": dom, "prompt_hash": ph, "rollout_idx": ri,
           "verifier_passed": bool(passed), "score": r.get("score"),
           "response_len_chars": len(sample["response"]),
           "response_length_tok": sample["response_length"]}
    resp = sample["response"]

    # C5: IF domain has no reasoning trajectory — deterministic path, no LLM call.
    if dom == "if":
        if passed:
            rec.update({
                "mode_source": "if_shortcut",
                "parsed": {"whole_clean": True, "has_clean_final": True, "keep_spans": [],
                           "mask_spans_summary": [], "reason": "IF domain: no reasoning trajectory to slice",
                           "risk_tags": []},
                "resolved": {"mode": "whole_clean", "kept_spans": [[0, len(resp)]],
                             "kept_frac": 1.0, "spans_status": []},
                "kept_preview": [resp[:400]],
                "error": None,
            })
        else:
            rec.update({
                "mode_source": "if_shortcut",
                "parsed": {"whole_clean": False, "has_clean_final": False, "keep_spans": [],
                           "mask_spans_summary": [], "reason": "IF domain: verifier failed",
                           "risk_tags": []},
                "resolved": {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0, "spans_status": []},
                "kept_preview": [],
                "error": None,
            })
        rec["revised_flag"] = False
        return rec

    # ---------------- pass 1 ----------------
    try:
        parsed = None
        raw1 = None
        for attempt in range(3):
            raw1 = call(SYSTEM_PROMPT_V3, build_user(sample))
            parsed = parse_json(raw1)
            if parsed is not None:
                break
            time.sleep(2 * (attempt + 1))
        if parsed is None:
            raise ValueError("pass1_parse_fail")
        resolved = resolve_spans(resp, parsed)
    except Exception as exc:
        rec.update({"parsed": None, "resolved": None, "error": f"pass1:{exc!r}"})
        return rec

    # C1: commit pass-1 result BEFORE pass-2 runs, so a pass-2 failure can't wipe it.
    rec.update({"parsed_pass1": parsed, "resolved_pass1": resolved,
                "parsed": parsed, "resolved": resolved,
                "kept_preview": [resp[a:b][:400] for a, b in resolved["kept_spans"]],
                "revised_flag": False, "revise_noop": False,
                "error": None, "pass2_error": None})

    # ---------------- pass 2 (audit + optional revise) ----------------
    if resolved["mode"] in ("spans_ok", "spans_partial", "fallback_last_to_end",
                            "fallback_tail_50pct", "whole_clean"):
        try:
            kept_text = reconstruct(resp, resolved["kept_spans"]) if resolved["kept_spans"] else ""
            audit = coherence_audit(sample, kept_text)
            rec["coherence_pass1"] = {k: audit[k] for k in
                                       ("coherent", "reaches_correct_answer", "severity",
                                        "issues", "missed_masks")}
            needs_revise = (audit.get("severity") == "major"
                            or audit.get("coherent") is False
                            or (audit.get("missed_masks") or []))
            if needs_revise:
                rev_raw = revise(sample, parsed, audit)
                rev_parsed = parse_json(rev_raw)
                # C2: detect no-op revise and try once more with higher temp
                if rev_parsed is not None and _same_spans(parsed, rev_parsed):
                    rec["revise_noop"] = True
                    rev_raw2 = call(SYSTEM_PROMPT_V3,
                                    build_user(sample) +
                                    "\n\n[PREVIOUS ANNOTATION was IDENTICAL to what you just returned. "
                                    "You MUST return a materially different annotation this time. "
                                    "If in doubt, choose action (c) DROP with has_clean_final=false.]\n"
                                    "[PRIOR INSUFFICIENT ANNOTATION]\n" + json.dumps(parsed, ensure_ascii=False) +
                                    "\n[AUDIT ISSUES]\n" + json.dumps(audit.get("issues", []) +
                                                                       audit.get("missed_masks", []),
                                                                       ensure_ascii=False),
                                    temperature=0.8)
                    rev_parsed2 = parse_json(rev_raw2)
                    if rev_parsed2 is not None and not _same_spans(parsed, rev_parsed2):
                        rev_parsed = rev_parsed2

                if rev_parsed is not None and not _same_spans(parsed, rev_parsed):
                    rev_resolved = resolve_spans(resp, rev_parsed)
                    rev_audit = coherence_audit(sample, reconstruct(resp, rev_resolved["kept_spans"])) \
                        if rev_resolved["mode"].startswith(("spans", "fallback", "whole_clean")) else None
                    rec["revised"] = {"parsed": rev_parsed, "resolved": rev_resolved,
                                      "coherence": ({k: rev_audit[k] for k in
                                                      ("coherent","reaches_correct_answer",
                                                       "severity","issues","missed_masks")}
                                                     if rev_audit else None)}
                    rec["revised_flag"] = True
                    # promote revised as the final annotation
                    rec["parsed"] = rev_parsed
                    rec["resolved"] = rev_resolved
                    rec["kept_preview"] = [resp[a:b][:400] for a, b in rev_resolved["kept_spans"]]
        except Exception as pass2_exc:
            # C1: pass-2 failure never wipes pass-1
            rec["pass2_error"] = repr(pass2_exc)

    # ---------------- C3: post-hoc sanity check on whole_clean ----------------
    if rec["resolved"] and rec["resolved"]["mode"] == "whole_clean":
        hits = whole_clean_suspect(resp)
        rec["whole_clean_suspect"] = hits or None
        # if there are strong signals AND we haven't already tried a revise, force one
        if hits and not rec["revised_flag"]:
            try:
                # synthesize a fake audit demanding a wrong-branch mask
                fake_audit = {"issues": [f"post-hoc heuristic found {list(hits.keys())} in the response "
                                          f"— whole_clean=true is likely wrong"],
                              "missed_masks": [f"heuristic hit: {k}={v}" for k, v in hits.items()]}
                rev_raw = revise(sample, rec["parsed"], fake_audit)
                rev_parsed = parse_json(rev_raw)
                if rev_parsed is not None and not _same_spans(rec["parsed"], rev_parsed):
                    rev_resolved = resolve_spans(resp, rev_parsed)
                    rev_audit = coherence_audit(sample, reconstruct(resp, rev_resolved["kept_spans"])) \
                        if rev_resolved["kept_spans"] else None
                    rec["revised"] = {"parsed": rev_parsed, "resolved": rev_resolved,
                                      "coherence": ({k: rev_audit[k] for k in
                                                      ("coherent","reaches_correct_answer",
                                                       "severity","issues","missed_masks")}
                                                     if rev_audit else None),
                                      "trigger": "whole_clean_suspect"}
                    rec["revised_flag"] = True
                    rec["parsed"] = rev_parsed
                    rec["resolved"] = rev_resolved
                    rec["kept_preview"] = [resp[a:b][:400] for a, b in rev_resolved["kept_spans"]]
            except Exception as post_exc:
                rec["post_hoc_error"] = repr(post_exc)

    return rec


def main():
    import torch
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = torch.load(ROLLOUT, map_location="cpu", weights_only=False)
    groups = {}
    for s in d["samples"]:
        ph = hashlib.md5(s["prompt"].encode()).hexdigest()[:8]
        groups.setdefault((s["metadata"]["domain"], ph), []).append(s)
    jobs = [(dom, ph, ri, s) for dom, hs in SELECTED.items() for ph in hs
            for ri, s in enumerate(groups[(dom, ph)])]
    print(f"v3: submitting {len(jobs)} jobs", flush=True)

    results = []
    with futures.ThreadPoolExecutor(max_workers=6) as ex:
        for i, f in enumerate(futures.as_completed({ex.submit(annotate, j): j for j in jobs}), 1):
            rec = f.result(); results.append(rec)
            res = rec.get("resolved") or {}
            coh = rec.get("coherence_pass1", {})
            print(f"[{i}/{len(jobs)}] {rec['domain']}/{rec['prompt_hash']}#{rec['rollout_idx']} "
                  f"pass={rec['verifier_passed']} mode={res.get('mode','ERR')} "
                  f"kept={(res.get('kept_frac') or 0):.2f} "
                  f"rev={rec.get('revised_flag')} noop={rec.get('revise_noop')} "
                  f"coh={coh.get('severity','-')} missed={len(coh.get('missed_masks') or [])} "
                  f"susp={'Y' if rec.get('whole_clean_suspect') else '-'} "
                  f"err={rec.get('error') or ''} p2={rec.get('pass2_error') or ''}",
                  flush=True)

    results.sort(key=lambda r: (r["domain"], r["prompt_hash"], r["rollout_idx"]))
    out = OUTDIR / "step1_sample24_annotations_v3.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
