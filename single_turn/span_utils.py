#!/usr/bin/env python3
"""Core utilities for grounded span selection over single-turn responses."""
from __future__ import annotations
import concurrent.futures as futures
import difflib
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

API = os.environ.get("ROSS_ENDPOINT", "http://localhost:8000/v1/chat/completions")
KEY = os.environ.get("ROSS_API_KEY", "")
MODEL = os.environ.get("ROSS_MODEL", "judge-model")

def api_headers():
    headers = {"Content-Type": "application/json"}
    if KEY:
        header = os.environ.get("ROSS_AUTH_HEADER", "Authorization")
        scheme = os.environ.get("ROSS_AUTH_SCHEME", "Bearer")
        headers[header] = f"{scheme} {KEY}".strip()
    return headers

ROLLOUT = Path(os.environ.get("ROSS_ROLLOUT_FILE", "./rollouts/sample.pt"))
OUTDIR = Path(os.environ.get("ROSS_OUTPUT_DIR", "./outputs"))

# 2 prompts/domain chosen from step 1 (md5(prompt)[:8]); mix of pass/fail.
SELECTED = {
    "math":   ["551c2083", "9ef1d658"],
    "codeio": ["d3cdca79", "974720f1"],
    "if":     ["45c3754e", "6861fd0e"],
}

SYSTEM_PROMPT = r"""You are a strict SEGMENT-LEVEL credit-assignment annotator. Your job is to turn one
reinforcement-learning rollout into masked supervised-fine-tuning (SFT) data.

BACKGROUND
An RL model was sampled on a problem. Its response often has this shape:
  correct setup -> a WRONG branch (wrong assumption/derivation/attempt) -> notices a
  contradiction -> self-correction ("wait, that's wrong...") -> a final CLEAN derivation
  that reaches the answer.
For SFT we must imitate token-by-token ONLY the parts that are worth learning. We must
MASK (not learn) the wrong exploration, abandoned attempts, and the self-correction
"trigger" text, because imitating them teaches the model to first go wrong.

YOUR TASK
Given the problem, the verifier verdict, the ground truth (when provided) and the
response, decide which contiguous spans of the response to KEEP (loss unmasked). All
text NOT covered by a keep-span is masked.

KEEP a span if it is:
  - correct problem setup / correct formulas that the final solution actually uses;
  - the final self-contained correct derivation and the final answer;
  - a genuine, intentional proof-by-contradiction (not a correction of a mistake).
MASK (do NOT put in keep-spans):
  - wrong assumptions, wrong derivations, arithmetic errors, dead ends;
  - abandoned attempts / discarded candidate solutions;
  - the self-correction trigger and hedging ("wait, this is wrong", "let me redo",
    "actually my previous...", "hmm, let me reconsider");
  - speculation that does not end up in the final solution;
  - for code: discarded code versions, hard-coded shortcuts, wrong algorithm talk.

IMPORTANT
  - Multiple keep-spans are allowed and expected (e.g. keep the correct setup at the
    start, mask a wrong middle, then keep the corrected final derivation). Preserve order.
  - The kept spans, read in order, should form a coherent correct solution on their own.
  - If the ENTIRE response is already a clean correct solution with no wrong branch, set
    whole_clean=true and keep_spans=[] (that means keep everything).
  - If the verifier FAILED, or there is no self-contained correct final solution, set
    has_clean_final=false and keep_spans=[] (nothing is safe to imitate).
  - Instruction-following / direct-answer responses usually have no reasoning trajectory:
    if the answer is coherent and satisfies the instructions, whole_clean=true.

ANCHORS
Each keep-span is given as {"start_anchor","end_anchor"}. Anchors MUST be copied
VERBATIM from the response, character-for-character (including LaTeX/markdown), be at
least 25 characters long, and be unique enough to locate by exact string search.
start_anchor = the first characters of the span to keep; end_anchor = the last
characters of the span to keep.

OUTPUT
Think briefly if needed, then end your reply with EXACTLY ONE fenced json block:
```json
{"whole_clean": false, "has_clean_final": true,
 "keep_spans": [{"start_anchor": "...", "end_anchor": "...", "why": "..."}],
 "mask_spans_summary": ["short description of each masked wrong/abandoned region"],
 "reason": "one short sentence",
 "risk_tags": ["self_correction","wrong_branch","abandoned_attempt","speculation","hardcode"]}
```

EXAMPLES (illustrative, abbreviated)

Example 1 - math, self-correction, MULTI-SPAN keep:
Response (abridged): "Set up P7 = (1/64)(P1 + L1 + 2L2 + ...). [correct] ... The line BC
is 3x+2y=84. [WRONG] ... So Sc<1.6, impossible, contradiction. Wait, earlier I
calculated 3x+2y=84. Recompute: 3x+4y=1680. [correction] ... So k=336, m=8. Answer: 344"
Expected:
```json
{"whole_clean": false, "has_clean_final": true,
 "keep_spans": [
   {"start_anchor": "Set up P7 = (1/64)(P1 + L1 + 2L2", "end_anchor": "targets when the die shows a vertex.", "why": "correct setup used by final solution"},
   {"start_anchor": "Recompute: 3x+4y=1680.", "end_anchor": "So k=336, m=8. Answer: 344", "why": "corrected final derivation and answer"}],
 "mask_spans_summary": ["wrong line equation 3x+2y=84 and the false contradiction it produced","self-correction trigger 'Wait, earlier I calculated'"],
 "reason": "correct setup then a wrong branch, self-corrected to the right answer",
 "risk_tags": ["self_correction","wrong_branch"]}
```

Example 2 - math, already clean:
Response: a single coherent derivation ending in one "Answer: 7".
```json
{"whole_clean": true, "has_clean_final": true, "keep_spans": [],
 "mask_spans_summary": [], "reason": "single coherent correct derivation", "risk_tags": []}
```

Example 3 - verifier FAILED, nothing to keep:
```json
{"whole_clean": false, "has_clean_final": false, "keep_spans": [],
 "mask_spans_summary": ["entire response reaches a wrong final answer"],
 "reason": "final answer is wrong, no self-contained correct solution", "risk_tags": ["wrong_reasoning"]}
```

Example 4 - codeio, discarded attempts:
Response tries matrix [[2,-1],[-1,2]] then says "the previous attempt failed", tries
[[1.5,0.5]...], finally settles on identity matrix and emits the final json input.
```json
{"whole_clean": false, "has_clean_final": true,
 "keep_spans": [{"start_anchor": "Let's stick to the Identity Matrix approach", "end_anchor": "\"vector_b\": [-2.1030175311241184, 2.0102705508663554]}}", "why": "final chosen approach and correct final input"}],
 "mask_spans_summary": ["discarded matrices [[2,-1],[-1,2]] and [[1.5,0.5]...] and precision self-correction"],
 "reason": "many abandoned attempts before the final correct input", "risk_tags": ["abandoned_attempt","self_correction"]}
```
"""


def build_user(sample) -> str:
    dom = sample["metadata"]["domain"]
    r = sample["reward"]
    if dom == "if":
        passed = bool(r.get("follow_all_instructions"))
        gt = "instructions: " + json.dumps(r.get("instruction_results", []), ensure_ascii=False)
    elif dom == "math":
        passed = r.get("score") == 1.0
        gt = "correct_answer = " + str((sample["metadata"].get("reward_spec") or {}).get("ground_truth"))
    else:  # codeio
        passed = r.get("score") == 1.0
        gt = "target_output = " + str(r.get("target_output"))
    return (f"[DOMAIN] {dom}\n[VERIFIER_PASSED] {passed}\n[GROUND_TRUTH] {gt}\n\n"
            f"[PROBLEM]\n{sample['prompt']}\n\n[RESPONSE]\n{sample['response']}\n\nAnnotate now.")


def call_judge(user: str, max_tokens: int = 32768) -> str:
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}],
               "stream": False, "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(API, data=json.dumps(payload, ensure_ascii=False).encode(),
                                 headers=api_headers(),
                                 method="POST")
    body = json.loads(urllib.request.urlopen(req, timeout=300).read())
    msg = body["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def parse_json(text: str):
    blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    cands = list(reversed(blocks)) + list(reversed(re.findall(r"\{.*\}", text, re.DOTALL)))
    for c in cands:
        try:
            o = json.loads(c)
            if isinstance(o, dict) and "keep_spans" in o:
                return o
        except Exception:
            continue
    return None


def _find(response: str, anchor: str):
    """Exact, then whitespace-normalized fuzzy locate. Returns (start,end) or None."""
    if not anchor:
        return None
    i = response.find(anchor)
    if i >= 0:
        return (i, i + len(anchor))
    # whitespace-normalized fuzzy: match token-collapsed anchor against windows
    norm = re.sub(r"\s+", " ", anchor).strip()
    if len(norm) >= 15:
        sm = difflib.SequenceMatcher(None, response, norm, autojunk=False)
        m = sm.find_longest_match(0, len(response), 0, len(norm))
        if m.size >= max(20, int(0.7 * len(norm))):
            return (m.a, m.a + m.size)
    return None


def resolve_spans(response: str, parsed: dict) -> dict:
    """Turn keep_spans anchors into char spans, applying fallbacks."""
    L = len(response)
    if parsed.get("whole_clean"):
        return {"mode": "whole_clean", "kept_spans": [[0, L]], "kept_frac": 1.0, "spans_status": []}
    if not parsed.get("has_clean_final") or not parsed.get("keep_spans"):
        return {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0, "spans_status": []}

    kept, status = [], []
    for sp in parsed["keep_spans"]:
        s = _find(response, sp.get("start_anchor", ""))
        e = _find(response, sp.get("end_anchor", ""))
        if s and e and e[1] > s[0]:
            kept.append([s[0], e[1]]); status.append("ok")
        elif s:
            kept.append([s[0], L]); status.append("end_missing->to_end")
        else:
            status.append("start_missing")

    if kept:
        kept.sort()
        merged = [kept[0]]
        for a, b in kept[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        frac = sum(b - a for a, b in merged) / L
        mode = "spans_ok" if all(x == "ok" for x in status) else "spans_partial"
        return {"mode": mode, "kept_spans": merged, "kept_frac": frac, "spans_status": status}

    # fallback: anchors all failed -> keep only last segment to end
    last = parsed["keep_spans"][-1]
    s = _find(response, last.get("start_anchor", ""))
    if s:
        return {"mode": "fallback_last_to_end", "kept_spans": [[s[0], L]],
                "kept_frac": (L - s[0]) / L, "spans_status": status}
    return {"mode": "fallback_tail_50pct", "kept_spans": [[L // 2, L]],
            "kept_frac": 0.5, "spans_status": status}


def annotate(job):
    dom, ph, ri, sample = job
    r = sample["reward"]
    passed = (r.get("follow_all_instructions") if dom == "if" else r.get("score") == 1.0)
    rec = {"domain": dom, "prompt_hash": ph, "rollout_idx": ri,
           "verifier_passed": bool(passed), "score": r.get("score"),
           "response_len_chars": len(sample["response"]),
           "response_length_tok": sample["response_length"]}
    for attempt in range(4):
        try:
            raw = call_judge(build_user(sample))
            parsed = parse_json(raw)
            if parsed is None:
                raise ValueError("parse_fail")
            resolved = resolve_spans(sample["response"], parsed)
            rec.update({"parsed": parsed, "resolved": resolved, "raw_judge": raw, "error": None})
            # store kept-text preview (heads only, to keep file small)
            rec["kept_preview"] = [sample["response"][a:b][:400] for a, b in resolved["kept_spans"]]
            return rec
        except Exception as exc:
            if attempt == 3:
                rec.update({"parsed": None, "resolved": None, "raw_judge": locals().get("raw", ""),
                            "error": repr(exc)})
                return rec
            time.sleep(2 * (attempt + 1))


def main():
    import torch
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = torch.load(ROLLOUT, map_location="cpu", weights_only=False)
    groups = {}
    for s in d["samples"]:
        dom = s["metadata"]["domain"]
        ph = hashlib.md5(s["prompt"].encode()).hexdigest()[:8]
        groups.setdefault((dom, ph), []).append(s)

    jobs = []
    for dom, hashes in SELECTED.items():
        for ph in hashes:
            for ri, s in enumerate(groups[(dom, ph)]):
                jobs.append((dom, ph, ri, s))
    print(f"submitting {len(jobs)} annotation jobs", flush=True)

    results = []
    with futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(annotate, j): j for j in jobs}
        for i, f in enumerate(futures.as_completed(futs), 1):
            rec = f.result()
            results.append(rec)
            print(f"[{i}/{len(jobs)}] {rec['domain']}/{rec['prompt_hash']}#{rec['rollout_idx']} "
                  f"pass={rec['verifier_passed']} "
                  f"mode={rec.get('resolved',{}).get('mode') if rec.get('resolved') else 'ERR'} "
                  f"kept={rec.get('resolved',{}).get('kept_frac') if rec.get('resolved') else '-'}",
                  flush=True)

    results.sort(key=lambda r: (r["domain"], r["prompt_hash"], r["rollout_idx"]))
    out = OUTDIR / "step1_sample24_annotations.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
