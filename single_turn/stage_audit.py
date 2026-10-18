#!/usr/bin/env python3
"""Independent audit and bounded repair for selected single-turn spans."""
from __future__ import annotations
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

try:
    import torch
except ImportError:  # JSONL runner does not require torch
    torch = None

# reuse v3's helpers (resolve_spans, reconstruct, IF-shortcut, whole_clean_suspect, etc.)
_spec = importlib.util.spec_from_file_location(
    "stage_core", str(Path(__file__).with_name("stage_core.py")))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)

API, KEY, MODEL = v3.API, v3.KEY, v3.MODEL
build_user = v3.build_user
parse_json = v3.parse_json
parse_obj = v3.parse_obj
resolve_spans = v3.resolve_spans
reconstruct = v3.reconstruct
whole_clean_suspect = v3.whole_clean_suspect

ROLLOUT_ROOT = Path(os.environ.get("ROSS_ROLLOUT_ROOT", "./rollouts"))
OUT_ROOT = Path(os.environ.get("ROSS_OUTPUT_DIR", "./outputs"))
RESULTS = OUT_ROOT / "scale_v4_rewardone_results.jsonl"
STATS   = OUT_ROOT / "scale_v4_rewardone_stats.json"

ROLLOUT_IDS = [50, 150, 250, 350, 450]
TARGET = {"math": 100, "codeio": 150}
RNG_SEED = 20260730  # same as v3 for direct comparison
MAX_WORKERS = 6


# =============================================================================
# Prompt fixes (Q1, Q2, Q3)
# =============================================================================

# Start from v3 and layer Q1 + Q2 on top of IMPORTANT.
_V3_SYS = v3.SYSTEM_PROMPT_V3

# Q1: kept spans must include derivation reaching the answer
_Q1_RULE = r"""

DERIVATION-COMPLETENESS (STRICT)
The concatenated kept spans MUST form a solution that not only STATES the answer
but SHOWS the derivation reaching that answer. If the kept text reads like:
    "Problem setup ... [big masked gap] ... Answer: X"
you have MASKED THE DERIVATION and the kept text is worthless as SFT signal.
Fix by KEEPING (not masking) the correct derivation region, even if that region
contains hedging like "Wait" or "Let me verify". The rule is: mask only WRONG
content; if a "Wait"/"Let me verify" segment leads to CORRECT calculations that
reach the final answer, it MUST be kept.

Concretely, for math responses the kept text should include:
  - the correct formula(s) used to reach the answer
  - the key algebraic / arithmetic steps (not just the final numeric line)
  - the arithmetic tying variables to the final numeric answer
For codeio (find-input-that-produces-output) the kept text should include:
  - the correct code-behavior analysis leading to the identified input
  - the verification step that shows the identified input produces the output

If the ONLY way to keep the derivation is to include a wrong branch (because it's
entangled), prefer setting has_clean_final=false over producing kept text with a
dangling "Answer: X".
"""

# Q2: strengthen the "don't over-mask" example
_Q2_EXAMPLE = r"""

Example 7 - CORRECT reasoning that HAPPENS to contain "Wait" — DO NOT MASK
Response derives a formula, then writes: "Let me verify: substituting x=3, we get
LHS = 27, RHS = 27. Wait, let me also check x=2: LHS = 12, RHS = 12. Confirmed."
The word "Wait" here is a SPEECH-LIKE FILLER that leads to CORRECT additional
verification — NOT a self-correction of a mistake. This entire block MUST be kept.
Only mask "Wait" when the sentence following it declares the previous work WRONG
(e.g. "Wait, that's wrong, let me redo"). Distinguish carefully:
  - "Wait, let me also check ..." → KEEP (extra verification)
  - "Wait, I made an error ..." → MASK (self-correction of mistake)
```json
{"whole_clean": false, "has_clean_final": true,
 "keep_spans": [{"start_anchor": "correct derivation starting text", "end_anchor": "Confirmed.", "why": "kept the correct derivation + its verification with 'Wait, let me also check' filler"}],
 "mask_spans_summary": [],
 "reason": "response is fully correct; 'Wait' here is verification filler, not error correction",
 "risk_tags": []}
```
"""

SYSTEM_PROMPT_V4 = (Path(__file__).parent / 'prompts' / 'annotation.txt').read_text(encoding="utf-8")


# Q3: verbatim requirement in audit
COHERENCE_SYSTEM_V4 = (Path(__file__).parent / 'prompts' / 'audit.txt').read_text(encoding="utf-8")


# =============================================================================
# Code fixes (F1-F5)
# =============================================================================

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


def coherence_audit(sample, kept_text: str) -> dict:
    dom = sample["metadata"]["domain"]
    gt = ("correct_answer = " + str((sample["metadata"].get("reward_spec") or {}).get("ground_truth"))
          if dom == "math" else "target_output = " + str(sample["reward"].get("target_output")))
    user = (f"[DOMAIN] {dom}\n[GROUND_TRUTH] {gt}\n\n[PROBLEM]\n{sample['prompt']}\n\n"
            f"[KEPT SOLUTION TO AUDIT (with masked gaps shown)]\n{kept_text}\n\n"
            f"[FULL ORIGINAL RESPONSE — for CHECK B]\n{sample['response']}\n\nAudit now.")
    raw = call(COHERENCE_SYSTEM_V4, user)
    parsed = parse_obj(raw, ("severity", "coherent", "missed_masks")) or {}
    return {"coherent": parsed.get("coherent"),
            "reaches_correct_answer": parsed.get("reaches_correct_answer"),
            "derivation_shown": parsed.get("derivation_shown"),
            "severity": parsed.get("severity", "unknown"),
            "issues": parsed.get("issues", []),
            "missed_masks": parsed.get("missed_masks", []),
            "raw": raw}


# F1: verify audit's missed_masks
def verify_missed_masks(response: str, kept_spans, raw_missed):
    """Return only the missed_masks that are (a) findable verbatim in response
    AND (b) actually located inside a kept span (not already masked)."""
    verified, rejected = [], []
    for m in raw_missed or []:
        if not isinstance(m, str) or len(m) < 20:
            rejected.append(("too_short", m[:60])); continue
        # try full then 60-char prefix
        query = m if len(m) <= 120 else m[:120]
        i = response.find(query)
        if i < 0:
            q40 = m[:40] if len(m) > 40 else m
            i = response.find(q40)
            if i < 0:
                rejected.append(("not_verbatim", m[:60])); continue
        # check span containment
        in_kept = any(a <= i < b for a, b in kept_spans)
        if not in_kept:
            rejected.append(("already_masked", m[:60])); continue
        verified.append(m)
    return verified, rejected


# F3: recompute severity from VERIFIED signals only
def verified_severity(issues, verified_missed, derivation_shown):
    if derivation_shown is False:
        return "major"
    if issues and verified_missed:
        return "major"
    if verified_missed or issues:
        return "minor"
    return "none"


# F2: force-revise prompt for anchor-failure specifically
def revise_anchor_failure(sample, prev_parsed) -> str:
    user = (build_user(sample) +
            "\n\n[PREVIOUS ANNOTATION — ANCHOR FAILURE]\n" +
            json.dumps(prev_parsed, ensure_ascii=False) +
            "\n\nYour previous keep_span anchors DID NOT resolve against the response text — "
            "the mask you intended did not apply, and everything was kept by fallback. "
            "You MUST produce a NEW annotation with DIFFERENT anchors. Rules:\n"
            "  - each start_anchor / end_anchor MUST be an exact character-for-character "
            "  copy of a UNIQUE substring in the response (>=25 chars, preferably >=40).\n"
            "  - copy anchors from actual response text, not from your description of it.\n"
            "  - do not include \\n or \\t escape sequences unless present literally.\n"
            "Alternatively, if there is no clean way to segment this response, set "
            "has_clean_final=false and keep_spans=[]. Return the json block only.")
    return call(SYSTEM_PROMPT_V4, user, temperature=0.4)


# F1 + F2 unified: general revise (mirrors v3 but calls v4 prompt)
def revise(sample, prev_parsed, audit) -> str:
    user = (build_user(sample) +
            "\n\n[PREVIOUS ANNOTATION — INSUFFICIENT]\n" + json.dumps(prev_parsed, ensure_ascii=False) +
            "\n\n[COHERENCE AUDIT ISSUES]\n" + json.dumps(audit.get("issues", []), ensure_ascii=False) +
            "\n\n[VERIFIED MISSED WRONG BRANCHES]\n" +
            json.dumps(audit.get("missed_masks_verified", []), ensure_ascii=False) +
            "\n\nYour PREVIOUS annotation was insufficient. Produce a NEW annotation that DIFFERS. "
            "Do NOT return identical keep_spans.\nActions:\n"
            "  (a) EXTEND — enlarge existing keep span(s) to include the correct DERIVATION "
            "     showing HOW the answer is reached (not just setup + answer);\n"
            "  (b) SHRINK — narrow keep span(s) that contain verified missed wrong content, "
            "     and/or explicitly mask the missed regions;\n"
            "  (c) DROP — if no clean self-contained correct solution can be extracted, set "
            "     has_clean_final=false and keep_spans=[].\n"
            "Return the json block only.")
    user += (Path(__file__).parent / "prompts" / "revision.txt").read_text(encoding="utf-8")
    return call(SYSTEM_PROMPT_V4, user, temperature=0.5)


# F4: trim duplicated tail Answer lines
_ANSWER_RE = re.compile(r"(?m)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
_CHAT_SUFFIX_RE = re.compile(r"<\|[^|>]*\|>|</\s*think\s*>", re.IGNORECASE)

def _norm_answer(s: str) -> str:
    """Strip chat suffixes like <|im_end|>, </think> so 'Answer: 42<|im_end|>'
    compares equal to 'Answer: 42'."""
    return _CHAT_SUFFIX_RE.sub("", s).strip()

def trim_duplicate_answer(response: str, kept_spans):
    """If the response ends with 'Answer: X ... Answer: X' (identical), and the
    last kept span contains both, trim so only the first Answer line is kept."""
    matches = list(_ANSWER_RE.finditer(response))
    if len(matches) < 2 or not kept_spans:
        return kept_spans, False
    last_two = matches[-2:]
    if _norm_answer(last_two[0].group(1)) != _norm_answer(last_two[1].group(1)):
        return kept_spans, False
    # find where the "second Answer" (duplicate) starts
    dup_start = last_two[1].start()
    first_ans_end = last_two[0].end()
    # trim last kept span if it extends past first_ans_end
    new_spans, trimmed = [], False
    for a, b in kept_spans:
        if b > dup_start and a < first_ans_end:
            # keep up to first_ans_end (drop the duplicate)
            new_end = min(b, first_ans_end + 2)  # +2 to keep trailing newline
            new_spans.append([a, new_end]); trimmed = True
        elif a >= dup_start:
            trimmed = True  # drop entirely
            continue
        else:
            new_spans.append([a, b])
    return new_spans, trimmed


# F5: substantive-derivation heuristic for long responses
_DERIV_HINT_RE = re.compile(
    r"(=\s*[-+]?\d|\\frac|\\sum|\\int|\\sqrt|\\cdot|\\times|\\div|"
    r"[+\-*/]\s*\d|\d\s*[+\-*/=]|"
    r"substituting|therefore|hence|so\s+we\s+get|thus|multiplying|dividing|solving)",
    re.IGNORECASE)

def looks_missing_derivation(response: str, kept_spans, min_len: int = 12000) -> bool:
    """Heuristic: for long responses with moderate mask, check that kept text
    has substantive derivation content, not just problem restatement + answer."""
    if len(response) < min_len:
        return False
    kept_text = "\n".join(response[a:b] for a, b in kept_spans)
    kept_len = len(kept_text)
    if kept_len == 0:
        return False
    frac = kept_len / len(response)
    if not (0.15 < frac < 0.55):
        return False
    # count derivation-like tokens per 1000 chars in kept text
    n_hint = len(_DERIV_HINT_RE.findall(kept_text))
    density = n_hint / max(1, kept_len / 1000)
    # empirical threshold: dense math should have > 3 hints per 1000 chars
    return density < 3.0


# =============================================================================
# Annotator (mirrors v3 with v4 fixes wired in)
# =============================================================================

def _same_spans(a_parsed, b_parsed):
    if a_parsed is None or b_parsed is None: return False
    ka = a_parsed.get("keep_spans") or []
    kb = b_parsed.get("keep_spans") or []
    if len(ka) != len(kb): return False
    for x, y in zip(ka, kb):
        if x.get("start_anchor") != y.get("start_anchor") or x.get("end_anchor") != y.get("end_anchor"):
            return False
    return a_parsed.get("whole_clean") == b_parsed.get("whole_clean") \
        and a_parsed.get("has_clean_final") == b_parsed.get("has_clean_final")


def annotate(job):
    dom, ph, si, sample = job
    r = sample["reward"]
    passed = (r.get("follow_all_instructions") if dom == "if" else r.get("score") == 1.0)
    rec = {"domain": dom, "prompt_hash": ph,
           "verifier_passed": bool(passed), "score": r.get("score"),
           "response_len_chars": len(sample["response"]),
           "response_length_tok": sample["response_length"]}
    resp = sample["response"]

    # IF shortcut (unchanged from v3)
    if dom == "if":
        if passed:
            rec.update({"mode_source": "if_shortcut",
                "parsed": {"whole_clean": True, "has_clean_final": True, "keep_spans": [],
                           "mask_spans_summary": [], "reason": "IF domain: no reasoning trajectory to slice",
                           "risk_tags": []},
                "resolved": {"mode": "whole_clean", "kept_spans": [[0, len(resp)]],
                             "kept_frac": 1.0, "spans_status": []},
                "kept_preview": [resp[:400]], "error": None})
        else:
            rec.update({"mode_source": "if_shortcut",
                "parsed": {"whole_clean": False, "has_clean_final": False, "keep_spans": [],
                           "mask_spans_summary": [], "reason": "IF domain: verifier failed",
                           "risk_tags": []},
                "resolved": {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0, "spans_status": []},
                "kept_preview": [], "error": None})
        rec["revised_flag"] = False
        return rec

    # ---------------- pass 1 ----------------
    try:
        parsed = None
        for attempt in range(3):
            raw1 = call(SYSTEM_PROMPT_V4, build_user(sample))
            parsed = parse_json(raw1)
            if parsed is not None: break
            time.sleep(2 * (attempt + 1))
        if parsed is None:
            raise ValueError("pass1_parse_fail")
        resolved = resolve_spans(resp, parsed)
    except Exception as exc:
        rec.update({"parsed": None, "resolved": None, "error": f"pass1:{exc!r}"})
        return rec

    rec.update({"parsed_pass1": parsed, "resolved_pass1": resolved,
                "parsed": parsed, "resolved": resolved,
                "kept_preview": [resp[a:b][:400] for a, b in resolved["kept_spans"]],
                "revised_flag": False, "revise_noop": False,
                "error": None, "pass2_error": None,
                "anchor_failure": False, "long_resp_over_mask": False})

    # F2: anchor-failure detection (before pass2)
    if resolved["mode"] == "spans_partial" and resolved["kept_frac"] >= 0.95:
        rec["anchor_failure"] = True
        try:
            rev_raw = revise_anchor_failure(sample, parsed)
            rev_parsed = parse_json(rev_raw)
            if rev_parsed is not None and not _same_spans(parsed, rev_parsed):
                rev_resolved = resolve_spans(resp, rev_parsed)
                rec["revised"] = {"parsed": rev_parsed, "resolved": rev_resolved,
                                  "trigger": "anchor_failure"}
                rec["revised_flag"] = True
                rec["parsed"] = rev_parsed
                rec["resolved"] = rev_resolved
                rec["kept_preview"] = [resp[a:b][:400] for a, b in rev_resolved["kept_spans"]]
                parsed = rev_parsed; resolved = rev_resolved
        except Exception as exc:
            rec["anchor_revise_error"] = repr(exc)

    # ---------------- pass 2 (audit + optional revise) ----------------
    if resolved["mode"] in ("spans_ok", "spans_partial", "fallback_last_to_end",
                            "fallback_tail_50pct", "whole_clean"):
        try:
            kept_text = reconstruct(resp, resolved["kept_spans"]) if resolved["kept_spans"] else ""
            audit = coherence_audit(sample, kept_text)

            # F1: verify missed_masks
            verified, rejected = verify_missed_masks(resp, resolved["kept_spans"],
                                                      audit.get("missed_masks", []))
            audit["missed_masks_verified"] = verified
            audit["missed_masks_rejected"] = [{"reason": r, "quote": q} for r, q in rejected]

            # F3: recompute severity from verified signals only
            audit["severity_verified"] = verified_severity(
                audit.get("issues"), verified, audit.get("derivation_shown"))

            rec["coherence_pass1"] = {k: audit[k] for k in
                                       ("coherent", "reaches_correct_answer", "derivation_shown",
                                        "severity", "severity_verified", "issues", "missed_masks",
                                        "missed_masks_verified", "missed_masks_rejected")}

            # F5: long-response missing-derivation check (only if not already flagged)
            if looks_missing_derivation(resp, resolved["kept_spans"]):
                rec["long_resp_over_mask"] = True
                if audit["severity_verified"] != "major":
                    audit["severity_verified"] = "major"
                if not any("derivation" in i.lower() for i in (audit.get("issues") or [])):
                    audit.setdefault("issues", []).append(
                        "long response with heavy mask: kept text appears to lack substantive "
                        "derivation content (mostly setup + conclusion)")

            # revise iff verified severity is major
            if audit["severity_verified"] == "major":
                rev_raw = revise(sample, parsed, audit)
                rev_parsed = parse_json(rev_raw)
                if rev_parsed is not None and _same_spans(parsed, rev_parsed):
                    rec["revise_noop"] = True
                    rev_raw2 = call(SYSTEM_PROMPT_V4,
                                    build_user(sample) +
                                    "\n\n[PRIOR annotation was IDENTICAL to what you returned. "
                                    "You MUST materially differ this time. If uncertain, choose DROP.]\n"
                                    "[PRIOR]\n" + json.dumps(parsed, ensure_ascii=False) +
                                    "\n[VERIFIED ISSUES]\n" +
                                    json.dumps((audit.get("issues") or []) + verified,
                                                ensure_ascii=False),
                                    temperature=0.8)
                    rev_parsed2 = parse_json(rev_raw2)
                    if rev_parsed2 is not None and not _same_spans(parsed, rev_parsed2):
                        rev_parsed = rev_parsed2

                if rev_parsed is not None and not _same_spans(parsed, rev_parsed):
                    rev_resolved = resolve_spans(resp, rev_parsed)
                    rec["revised"] = {"parsed": rev_parsed, "resolved": rev_resolved,
                                      "trigger": "verified_major"}
                    rec["revised_flag"] = True
                    rec["parsed"] = rev_parsed
                    rec["resolved"] = rev_resolved
                    rec["kept_preview"] = [resp[a:b][:400] for a, b in rev_resolved["kept_spans"]]
        except Exception as pass2_exc:
            rec["pass2_error"] = repr(pass2_exc)

    # ---------------- post-hoc sanity: whole_clean_suspect ----------------
    if rec.get("resolved") and rec["resolved"]["mode"] == "whole_clean":
        hits = whole_clean_suspect(resp)
        rec["whole_clean_suspect"] = hits or None
        if hits and not rec.get("revised_flag"):
            try:
                fake_audit = {"issues": [f"heuristic detected {list(hits.keys())} in whole_clean response"],
                              "missed_masks_verified": [f"hit: {k}" for k in hits]}
                rev_raw = revise(sample, rec["parsed"], fake_audit)
                rev_parsed = parse_json(rev_raw)
                if rev_parsed is not None and not _same_spans(rec["parsed"], rev_parsed):
                    rev_resolved = resolve_spans(resp, rev_parsed)
                    rec["revised"] = {"parsed": rev_parsed, "resolved": rev_resolved,
                                      "trigger": "whole_clean_suspect"}
                    rec["revised_flag"] = True
                    rec["parsed"] = rev_parsed
                    rec["resolved"] = rev_resolved
                    rec["kept_preview"] = [resp[a:b][:400] for a, b in rev_resolved["kept_spans"]]
            except Exception as post_exc:
                rec["post_hoc_error"] = repr(post_exc)

    # ---------------- F4: post-hoc duplicate-Answer trim ----------------
    if rec.get("resolved") and rec["resolved"]["kept_spans"]:
        trimmed_spans, was_trimmed = trim_duplicate_answer(resp, rec["resolved"]["kept_spans"])
        if was_trimmed:
            L = len(resp)
            new_frac = sum(b - a for a, b in trimmed_spans) / L
            rec["resolved"]["kept_spans"] = trimmed_spans
            rec["resolved"]["kept_frac"] = new_frac
            rec["duplicate_answer_trimmed"] = True
            rec["kept_preview"] = [resp[a:b][:400] for a, b in trimmed_spans]

    return rec


# =============================================================================
# Sampling + driver (mirrors v3_scale)
# =============================================================================

def passed(sample):
    dom = sample["metadata"]["domain"]
    r = sample["reward"]
    if dom == "if": return bool(r.get("follow_all_instructions"))
    return r.get("score") == 1.0


def collect_reward_one(rollout_ids):
    pool = {"math": [], "codeio": [], "if": []}
    for rid in rollout_ids:
        fp = ROLLOUT_ROOT / f"{rid}.pt"
        if not fp.exists(): print(f"[collect] skip missing {fp.name}", flush=True); continue
        d = torch.load(fp, map_location="cpu", weights_only=False)
        n = {"math": 0, "codeio": 0, "if": 0}
        for si, s in enumerate(d["samples"]):
            if not passed(s): continue
            dom = s["metadata"]["domain"]
            ph = hashlib.md5(s["prompt"].encode()).hexdigest()[:8]
            pool[dom].append((rid, si, ph, s))
            n[dom] += 1
        print(f"[collect] rollout {rid}: math={n['math']} codeio={n['codeio']} if={n['if']}", flush=True)
        del d
    return pool


def sample_pool(pool, targets, seed):
    rng = random.Random(seed)
    picks = []
    for dom, cap in targets.items():
        arr = pool[dom]
        chosen = arr if len(arr) <= cap else rng.sample(arr, cap)
        picks.extend(chosen)
        print(f"[sample] {dom}: picked {len(chosen)} of {len(arr)}", flush=True)
    picks.extend(pool["if"])
    print(f"[sample] if: added {len(pool['if'])} (shortcut)", flush=True)
    return picks


def load_processed(path):
    if not path.exists(): return set()
    done = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                done.add((r["rollout_id"], r["sample_idx"]))
            except Exception: pass
    return done


def annotate_and_tag(job_and_rid):
    (dom, ph, si, sample), rid = job_and_rid
    rec = annotate((dom, ph, si, sample))
    rec["rollout_id"] = rid
    rec["sample_idx"] = si
    return rec


def summarise(results):
    import collections
    by_dom = collections.defaultdict(lambda: collections.Counter())
    kept = collections.defaultdict(list)
    special = collections.Counter()
    for r in results:
        dom = r["domain"]
        mode = (r.get("resolved") or {}).get("mode","ERR")
        by_dom[dom][mode] += 1
        kf = (r.get("resolved") or {}).get("kept_frac")
        if kf is not None: kept[dom].append(kf)
        if r.get("anchor_failure"): special["anchor_failure"] += 1
        if r.get("long_resp_over_mask"): special["long_resp_over_mask"] += 1
        if r.get("duplicate_answer_trimmed"): special["duplicate_answer_trimmed"] += 1
        if r.get("revise_noop"): special["revise_noop"] += 1
        if r.get("whole_clean_suspect"): special["whole_clean_suspect"] += 1
    out = {"per_domain": {}}
    for dom, c in by_dom.items():
        ks = kept[dom]
        mean = (sum(ks)/len(ks)) if ks else 0.0
        usable = sum(1 for k in ks if k > 0) / len(ks) if ks else 0.0
        span_frac = (c.get("spans_ok",0) + c.get("spans_partial",0)) / sum(c.values()) if c else 0.0
        out["per_domain"][dom] = {
            "n": sum(c.values()), "modes": dict(c),
            "mean_kept_frac": round(mean, 3), "usable_frac": round(usable, 3),
            "whole_clean_frac": round(c.get("whole_clean", 0) / sum(c.values()), 3) if c else 0.0,
            "spans_frac": round(span_frac, 3),
            "dropped_frac": round(c.get("dropped", 0) / sum(c.values()), 3) if c else 0.0,
        }
    out["special_flags"] = dict(special)
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[start-v4] target={TARGET} + all-IF | rollouts={ROLLOUT_IDS} | seed={RNG_SEED}", flush=True)

    pool = collect_reward_one(ROLLOUT_IDS)
    picks = sample_pool(pool, TARGET, RNG_SEED)
    print(f"[plan] total picks: {len(picks)} (non-IF: {sum(1 for p in picks if p[3]['metadata']['domain']!='if')})", flush=True)

    done = load_processed(RESULTS)
    print(f"[resume] {len(done)} already processed, skipping", flush=True)
    remaining = [p for p in picks if (p[0], p[1]) not in done]
    print(f"[plan] {len(remaining)} jobs to run", flush=True)

    jobs = [((s["metadata"]["domain"], ph, si, s), rid) for rid, si, ph, s in remaining]
    t0 = time.time()
    with RESULTS.open("a", encoding="utf-8") as fout:
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(annotate_and_tag, j): j for j in jobs}
            for i, f in enumerate(futures.as_completed(futs), 1):
                try:
                    rec = f.result()
                except Exception as exc:
                    (dom, ph, si, _), rid = futs[f]
                    rec = {"rollout_id": rid, "sample_idx": si, "domain": dom,
                           "prompt_hash": ph, "resolved": None, "parsed": None,
                           "error": f"outer:{exc!r}"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                res = rec.get("resolved") or {}
                coh = rec.get("coherence_pass1") or {}
                el = time.time() - t0
                flags = []
                if rec.get("anchor_failure"): flags.append("ANCH")
                if rec.get("long_resp_over_mask"): flags.append("LONG")
                if rec.get("duplicate_answer_trimmed"): flags.append("DUP")
                if rec.get("whole_clean_suspect"): flags.append("WCS")
                if rec.get("revise_noop"): flags.append("NOOP")
                flag_str = "|".join(flags) if flags else "-"
                print(f"[{i}/{len(jobs)}] r{rec.get('rollout_id')}#{rec.get('sample_idx')} "
                      f"{rec['domain']}/{rec.get('prompt_hash')} "
                      f"mode={res.get('mode','ERR')} kept={(res.get('kept_frac') or 0):.2f} "
                      f"rev={rec.get('revised_flag')} "
                      f"sev={coh.get('severity_verified','-')}(raw:{coh.get('severity','-')}) "
                      f"vmiss={len(coh.get('missed_masks_verified') or [])}/{len(coh.get('missed_masks') or [])} "
                      f"flag={flag_str} err={rec.get('error') or ''} p2={rec.get('pass2_error') or ''} "
                      f"[{el:.0f}s]", flush=True)

    print(f"\n[all-done] {time.time()-t0:.0f}s", flush=True)
    all_results = [json.loads(l) for l in RESULTS.open()]
    stats = summarise(all_results)
    stats["config"] = {"rollout_ids": ROLLOUT_IDS, "target": TARGET, "seed": RNG_SEED,
                       "max_workers": MAX_WORKERS, "total_samples": len(all_results),
                       "version": "stage_audit"}
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[stats] wrote {STATS}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
