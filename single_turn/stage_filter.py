#!/usr/bin/env python3
"""Protocol-violation filtering before single-turn span annotation."""
from __future__ import annotations
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path
try:
    import torch
except ImportError:  # JSONL runner does not require torch
    torch = None

# reuse v4 (which reuses v3 which reuses v1)
_spec = importlib.util.spec_from_file_location(
    "stage_audit", str(Path(__file__).with_name("stage_audit.py")))
v4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v4)

ROLLOUT_ROOT = Path(os.environ.get("ROSS_ROLLOUT_ROOT", "./rollouts"))
OUT_ROOT = Path(os.environ.get("ROSS_OUTPUT_DIR", "./outputs"))
RESULTS = OUT_ROOT / "scale_v5_rewardone_results.jsonl"
STATS   = OUT_ROOT / "scale_v5_rewardone_stats.json"

ROLLOUT_IDS = [50, 150, 250, 350, 450]
TARGET = {"math": 100, "codeio": 150}
RNG_SEED = 20260730
MAX_WORKERS = 6


# =============================================================================
# The one and only v5 addition: nothink violation filter
# =============================================================================

THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

def detect_nothink_violation(response: str):
    """Return violation info dict if response contains any <think> or </think>
    tag (either indicates the model broke the no_think protocol).

    Returns None if response is clean."""
    matches = list(THINK_TAG_RE.finditer(response))
    if not matches:
        return None
    # locate the LAST </think> — real answer (if any) is only after it
    last_close = None
    for m in matches:
        if m.group(0).lower() == "</think>":
            last_close = m
    return {
        "violation": True,
        "n_tags": len(matches),
        "tags_found": sorted({m.group(0) for m in matches}),
        "junk_end": last_close.end() if last_close else matches[-1].end(),
        "junk_frac": (last_close.end() / len(response)) if last_close else (matches[-1].end() / len(response)),
    }


def annotate(job):
    """v5 annotate: first check for nothink violation, drop if found; else delegate to v4."""
    dom, ph, si, sample = job
    resp = sample["response"]

    # NEW v5: nothink violation short-circuit
    v = detect_nothink_violation(resp)
    if v is not None:
        r = sample["reward"]
        passed = (r.get("follow_all_instructions") if dom == "if" else r.get("score") == 1.0)
        return {
            "domain": dom, "prompt_hash": ph,
            "verifier_passed": bool(passed), "score": r.get("score"),
            "response_len_chars": len(resp),
            "response_length_tok": sample["response_length"],
            "mode_source": "nothink_violation_drop",
            "nothink_violation": v,
            "parsed": {"whole_clean": False, "has_clean_final": False, "keep_spans": [],
                       "mask_spans_summary": [f"nothink protocol violation: response contains {v['tags_found']} tags "
                                              f"({v['junk_frac']*100:.0f}% junk before last </think>)"],
                       "reason": "nothink violation — dropped per training protocol",
                       "risk_tags": ["nothink_violation"]},
            "resolved": {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0, "spans_status": []},
            "kept_preview": [], "revised_flag": False, "error": None,
        }

    # otherwise delegate to v4 annotate
    return v4.annotate(job)


# ---------- sampling + driver (identical to v4_scale) ----------
def passed(sample):
    dom = sample["metadata"]["domain"]; r = sample["reward"]
    if dom == "if": return bool(r.get("follow_all_instructions"))
    return r.get("score") == 1.0


def collect_reward_one(rollout_ids):
    pool = {"math": [], "codeio": [], "if": []}
    for rid in rollout_ids:
        fp = ROLLOUT_ROOT / f"{rid}.pt"
        if not fp.exists(): print(f"[collect] skip {fp.name}", flush=True); continue
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
                r = json.loads(line); done.add((r["rollout_id"], r["sample_idx"]))
            except Exception: pass
    return done


def annotate_and_tag(job_and_rid):
    (dom, ph, si, sample), rid = job_and_rid
    rec = annotate((dom, ph, si, sample))
    rec["rollout_id"] = rid; rec["sample_idx"] = si
    return rec


def summarise(results):
    import collections
    by_dom = collections.defaultdict(lambda: collections.Counter())
    kept = collections.defaultdict(list)
    special = collections.Counter()
    for r in results:
        dom = r["domain"]
        mode = (r.get("resolved") or {}).get("mode", "ERR")
        by_dom[dom][mode] += 1
        kf = (r.get("resolved") or {}).get("kept_frac")
        if kf is not None: kept[dom].append(kf)
        if r.get("nothink_violation"): special["nothink_violation_dropped"] += 1
        if r.get("anchor_failure"): special["anchor_failure"] += 1
        if r.get("long_resp_over_mask"): special["long_resp_over_mask"] += 1
        if r.get("duplicate_answer_trimmed"): special["duplicate_answer_trimmed"] += 1
        if r.get("whole_clean_suspect"): special["whole_clean_suspect"] += 1
    out = {"per_domain": {}}
    for dom, c in by_dom.items():
        ks = kept[dom]; mean = (sum(ks)/len(ks)) if ks else 0.0
        usable = sum(1 for k in ks if k > 0) / len(ks) if ks else 0.0
        out["per_domain"][dom] = {"n": sum(c.values()), "modes": dict(c),
            "mean_kept_frac": round(mean, 3), "usable_frac": round(usable, 3),
            "whole_clean_frac": round(c.get("whole_clean", 0)/sum(c.values()), 3) if c else 0.0,
            "spans_frac": round((c.get("spans_ok",0)+c.get("spans_partial",0))/sum(c.values()), 3) if c else 0.0,
            "dropped_frac": round(c.get("dropped", 0)/sum(c.values()), 3) if c else 0.0}
    out["special_flags"] = dict(special)
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[start-v5] target={TARGET} + all-IF | rollouts={ROLLOUT_IDS} | seed={RNG_SEED}", flush=True)
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
                try: rec = f.result()
                except Exception as exc:
                    (dom, ph, si, _), rid = futs[f]
                    rec = {"rollout_id": rid, "sample_idx": si, "domain": dom, "prompt_hash": ph,
                           "resolved": None, "parsed": None, "error": f"outer:{exc!r}"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                res = rec.get("resolved") or {}
                el = time.time() - t0
                nv = "NOTHINK-DROP" if rec.get("nothink_violation") else "-"
                print(f"[{i}/{len(jobs)}] r{rec.get('rollout_id')}#{rec.get('sample_idx')} "
                      f"{rec['domain']}/{rec.get('prompt_hash')} mode={res.get('mode','ERR')} "
                      f"kept={(res.get('kept_frac') or 0):.2f} filter={nv} [{el:.0f}s]", flush=True)
    print(f"\n[all-done] {time.time()-t0:.0f}s", flush=True)
    all_results = [json.loads(l) for l in RESULTS.open()]
    stats = summarise(all_results)
    stats["config"] = {"rollout_ids": ROLLOUT_IDS, "target": TARGET, "seed": RNG_SEED,
                       "max_workers": MAX_WORKERS, "total_samples": len(all_results), "version": "stage_filter"}
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[stats] wrote {STATS}"); print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
