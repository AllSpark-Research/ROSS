#!/usr/bin/env python3
"""Final single-turn annotation and deterministic acceptance pipeline.

The annotator combines proposal, independent audit, bounded revision, exact
anchor validation, final-answer preservation, and conservative rejection when
a clean self-contained target cannot be recovered from the saved response.
"""
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

# reuse v5 (which reuses v4 which reuses v3 which reuses v1)
_spec = importlib.util.spec_from_file_location(
    "stage_filter", str(Path(__file__).with_name("stage_filter.py")))
v5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v5)
v4 = v5.v4
v3 = v4.v3
seg = v3.seg

# reuse most of v4
call = v4.call
build_user = v4.build_user
parse_json = v4.parse_json
parse_obj = v4.parse_obj
resolve_spans = v4.resolve_spans
reconstruct = v4.reconstruct
whole_clean_suspect = v4.whole_clean_suspect
_legacy_coherence_audit = v4.coherence_audit
_guard_spec = importlib.util.spec_from_file_location(
    "revision_guard", Path(__file__).with_name("revision_guard.py"))
guard = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(guard)


def coherence_audit(sample, kept_text):
    audit = _legacy_coherence_audit(sample, kept_text)
    obj = parse_obj(audit.get("raw", ""), ("severity", "coherent", "missed_masks")) or {}
    audit["checked_claims"] = obj.get("checked_claims", [])
    audit["conflict_pairs"] = obj.get("conflict_pairs", [])
    return audit
verify_missed_masks = v4.verify_missed_masks
verified_severity = v4.verified_severity
looks_missing_derivation = v4.looks_missing_derivation
revise = v4.revise
revise_anchor_failure = v4.revise_anchor_failure
trim_duplicate_answer = v4.trim_duplicate_answer
_same_spans = v4._same_spans
SYSTEM_PROMPT_V4 = v4.SYSTEM_PROMPT_V4

ROLLOUT_ROOT = Path(os.environ.get("ROSS_ROLLOUT_ROOT", "./rollouts"))
OUT_ROOT = Path(os.environ.get("ROSS_OUTPUT_DIR", "./outputs"))
RESULTS = OUT_ROOT / "single_turn_annotations.jsonl"
STATS   = OUT_ROOT / "single_turn_annotation_stats.json"

ROLLOUT_IDS = [50, 150, 250, 350, 450]
TARGET = {"math": 100, "codeio": 150}
RNG_SEED = 20260730
MAX_WORKERS = 6


# =============================================================================
# Final quality-gate helpers
# =============================================================================

# --- I1: extended anchor_failure detection ---
def is_hidden_anchor_failure(resolved, parsed):
    """spans_ok+kept≈1.0 with non-empty mask_spans_summary means judge intended
    to mask but the resolved span covers everything anyway (anchors likely
    snapped to response boundaries)."""
    mode = resolved.get("mode")
    kept_frac = resolved.get("kept_frac", 0)
    if mode == "spans_ok" and kept_frac >= 0.98:
        if not parsed.get("whole_clean"):
            mask_summary = parsed.get("mask_spans_summary") or []
            if mask_summary:
                return True
    return False


def is_standard_anchor_failure(resolved):
    """The v4/v5 detection: spans_partial + kept≥0.95 means anchor didn't
    resolve at all and fallback kept everything."""
    return (resolved.get("mode") == "spans_partial"
            and resolved.get("kept_frac", 0) >= 0.95)


def anchor_failure_kind(resolved, parsed):
    """Returns 'standard' / 'hidden' / None."""
    if is_standard_anchor_failure(resolved): return "standard"
    if is_hidden_anchor_failure(resolved, parsed): return "hidden"
    return None


# --- I2: final-answer preservation check ---
_MATH_ANSWER_RE = re.compile(r"(?im)^[ \t]*(?P<answer>(?:final[ \t]+)?answer[ \t]*:[ \t]*[^\r\n]+?)(?:[ \t]*<\|(?:im_end|endoftext)\|>)?[ \t]*\r?$")
_CODEIO_JSON_RE = re.compile(r'```json\s*\{[^`]*?\}\s*```', re.DOTALL)
_CODEIO_JSON_LOOSE_RE = re.compile(r'\{\s*"(?:input|output)"\s*:.+?\}(?:\s*```)?', re.DOTALL)

def codeio_answer_spans(response):
    """Complete JSON answer payloads; Markdown fences are not answer content."""
    decoder = json.JSONDecoder()
    spans = []
    for match in re.finditer(r'\{\s*"(?:input|output)"\s*:', response):
        try:
            value, length = decoder.raw_decode(response[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and ('input' in value or 'output' in value):
            spans.append((match.start(), match.start() + length))
    return spans


def find_final_answer_pos(response, domain):
    """Return (start, end) of the FINAL answer text in the response.
    Returns None if no clear final answer marker is found."""
    if domain == "math":
        matches = list(_MATH_ANSWER_RE.finditer(response))
        if matches:
            return matches[-1].span("answer")
    elif domain == "codeio":
        spans = codeio_answer_spans(response)
        if spans:
            return spans[-1]
    return None


def kept_covers_final_answer(response, kept_spans, domain):
    """True iff at least one complete math answer (or CodeIO marker) is kept.

    (Fixed: earlier version only checked the LAST answer marker's position, which
    fails when trim_duplicate_answer has trimmed a tail duplicate — the LAST
    marker becomes the trimmed-off duplicate. The correct check is 'kept covers
    ANY answer marker'.)
    Returns True (fail-open) if no answer marker can be located."""
    if domain == "math":
        matches = list(_MATH_ANSWER_RE.finditer(response))
    elif domain == "codeio":
        answers = codeio_answer_spans(response)
        if not answers:
            return True
        return any(any(a <= start and end <= b for a, b in kept_spans)
                   for start, end in answers)
    else:
        return True
    if not matches:
        return True
    if domain == "math":
        # Whitespace before Answer is not part of the answer. Require the
        # entire answer text, not just the marker start, to remain supervised.
        return any(any(a <= m.start("answer") and m.end("answer") <= b
                       for a, b in kept_spans) for m in matches)
    return any(any(a <= m.start() and m.end() <= b for a, b in kept_spans) for m in matches)


# --- I3: thin kept text detection ---
_DERIV_HINT_RE = re.compile(
    r"(=\s*[-+]?\d|\\frac|\\sum|\\int|\\sqrt|\\cdot|\\times|\\div|"
    r"[+\-*/]\s*\d|\d\s*[+\-*/=]|"
    r"substituting|therefore|hence|so\s+we\s+get|thus|multiplying|dividing|solving)",
    re.IGNORECASE)

_CODE_HINT_RE = re.compile(
    r"```|\bdef\s|\breturn\b|\binput\s*\(|\bprint\s*\(|"
    r"=\s*['\"]|=\s*\[|=\s*\{|\bfor\s+\w+\s+in\b|\bwhile\s+\w+|\bif\s+\w+")

def is_thin_kept_text(response, kept_spans, domain, min_resp_len=2000):
    """kept_frac<0.25 AND kept text has too few derivation markers → useless SFT.

    Returns True (should drop) or False (keep).
    Guards: only fires on long responses (>min_resp_len); ignores whole_clean/dropped."""
    if not kept_spans or len(response) < min_resp_len:
        return False
    kept_len = sum(b - a for a, b in kept_spans)
    if kept_len == 0:
        return False
    frac = kept_len / len(response)
    if frac >= 0.25:
        return False
    kept_text = "\n".join(response[a:b] for a, b in kept_spans)
    if domain == "math":
        hints = len(_DERIV_HINT_RE.findall(kept_text))
        return hints < 5
    elif domain == "codeio":
        hints = len(_CODE_HINT_RE.findall(kept_text))
        return hints < 3
    return False


# --- helper: try a targeted revise for a hidden anchor failure ---
def revise_hidden_anchor_failure(sample, prev_parsed) -> str:
    """Same as v4's revise_anchor_failure but with a message tailored to the
    'anchors landed at boundaries' pattern."""
    user = (build_user(sample) +
            "\n\n[PREVIOUS ANNOTATION — ANCHORS SNAPPED TO BOUNDARIES]\n" +
            json.dumps(prev_parsed, ensure_ascii=False) +
            "\n\nYou intended to mask the content listed in mask_spans_summary, "
            "but your keep_span anchors matched positions covering the ENTIRE response, "
            "so nothing was actually masked.\n"
            "Please rewrite the annotation with anchors that resolve to precisely "
            "the correct boundaries, so mask regions actually get excluded. Rules:\n"
            "  - each start_anchor / end_anchor MUST be an exact character-for-character "
            "copy of a UNIQUE substring in the response (≥25 chars, preferably ≥40)\n"
            "  - copy anchors from actual response text, not from your description of it\n"
            "  - do not include escape sequences unless literal\n"
            "Alternatively, if the wrong content is not cleanly separable, set "
            "has_clean_final=false and keep_spans=[]. Return the json block only.")
    return call(SYSTEM_PROMPT_V4, user, temperature=0.4)


# --- helper: apply revise with post-check, revert on final-answer loss ---
def try_revise_with_answer_check(sample, resp, dom, prev_parsed, prev_resolved,
                                  audit_or_reason, trigger_label, attempt_log=None):
    """Run a revise attempt. If revise produced a materially different result AND
    the result still covers the final answer, return (new_parsed, new_resolved, True).
    If revise lost the answer or was identical, return (prev_parsed, prev_resolved, False).
    """
    trace = {"trigger": trigger_label}
    if attempt_log is not None:
        attempt_log.append(trace)
    try:
        if trigger_label.startswith("anchor_"):
            if trigger_label == "anchor_standard":
                rev_raw = revise_anchor_failure(sample, prev_parsed)
            else:
                rev_raw = revise_hidden_anchor_failure(sample, prev_parsed)
        else:
            rev_raw = revise(sample, prev_parsed, audit_or_reason)
        trace["raw"] = rev_raw
        rev_parsed, parse_error = guard.parse_annotation(rev_raw)
        trace["parse_error"] = parse_error
        if parse_error:
            return prev_parsed, prev_resolved, False, parse_error
        if _same_spans(prev_parsed, rev_parsed):
            return prev_parsed, prev_resolved, False, "revision_unchanged"
        rev_resolved = resolve_spans(resp, rev_parsed)
        trace.update(parsed=rev_parsed, resolved=rev_resolved)
        errors = guard.candidate_errors(resp, rev_parsed, rev_resolved,
                                        kept_covers_final_answer, dom)
        trace["validation_errors"] = errors
        if errors:
            return prev_parsed, prev_resolved, False, "revision_validation:" + ",".join(errors)
        if isinstance(audit_or_reason, dict):
            findings, _ = guard.audit_evidence(resp, prev_resolved.get("kept_spans", []), audit_or_reason)
            if any(guard.intersects(rev_resolved.get("kept_spans", []), f["interval"]) for f in findings):
                return prev_parsed, prev_resolved, False, "confirmed_error_still_supervised"
        # I2: check final answer preserved
        if rev_resolved.get("kept_spans"):
            if not kept_covers_final_answer(resp, rev_resolved["kept_spans"], dom):
                return prev_parsed, prev_resolved, False, "revise_lost_final_answer"
        # otherwise (dropped or covers answer): accept
        return rev_parsed, rev_resolved, True, "ok"
    except Exception as exc:
        return prev_parsed, prev_resolved, False, f"revise_exc:{exc!r}"


# =============================================================================
# Final annotation pass — clean rewrite so decision points are all explicit
# =============================================================================

def _annotate_legacy(job, use_if_shortcut=False):
    dom, ph, si, sample = job
    r = sample["reward"]
    passed = (r.get("follow_all_instructions") if dom == "if" else r.get("score") == 1.0)
    resp = sample["response"]
    rec = {"domain": dom, "prompt_hash": ph,
           "verifier_passed": bool(passed), "score": r.get("score"),
           "response_len_chars": len(resp),
           "response_length_tok": sample["response_length"]}

    # ---------------- Step 0: v5 nothink filter (highest priority) ----------------
    viol = v5.detect_nothink_violation(resp)
    if viol is not None:
        rec.update({
            "mode_source": "nothink_violation_drop",
            "nothink_violation": viol,
            "parsed": {"whole_clean": False, "has_clean_final": False, "keep_spans": [],
                       "mask_spans_summary": [f"nothink protocol violation: {viol['tags_found']}, "
                                              f"{viol['junk_frac']*100:.0f}% junk"],
                       "reason": "nothink violation — dropped", "risk_tags": ["nothink_violation"]},
            "resolved": {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0, "spans_status": []},
            "kept_preview": [], "revised_flag": False, "error": None,
        })
        return rec

    # ---------------- Step 1: IF shortcut (skippable) ----------------
    if dom == "if" and use_if_shortcut:
        if passed:
            rec.update({"mode_source": "if_shortcut",
                "parsed": {"whole_clean": True, "has_clean_final": True, "keep_spans": [],
                           "mask_spans_summary": [], "reason": "IF: direct answer",
                           "risk_tags": []},
                "resolved": {"mode": "whole_clean", "kept_spans": [[0, len(resp)]],
                             "kept_frac": 1.0, "spans_status": []},
                "kept_preview": [resp[:400]], "revised_flag": False, "error": None})
        else:
            rec.update({"mode_source": "if_shortcut",
                "parsed": {"whole_clean": False, "has_clean_final": False, "keep_spans": [],
                           "mask_spans_summary": [], "reason": "IF: verifier failed",
                           "risk_tags": []},
                "resolved": {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0,
                             "spans_status": []},
                "kept_preview": [], "revised_flag": False, "error": None})
        return rec

    # ---------------- Step 2: pass 1 (math / codeio) ----------------
    try:
        parsed = None
        for attempt in range(3):
            raw1 = call(SYSTEM_PROMPT_V4, build_user(sample))
            parsed, parse_error = guard.parse_annotation(raw1)
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
                "revised_flag": False, "revise_history": [],
                "error": None, "pass2_error": None,
                "anchor_failure": None, "long_resp_over_mask": False})

    # ---------------- Step 3: I1 — anchor failure detection (both kinds) ----------------
    kind = anchor_failure_kind(resolved, parsed)
    if kind is not None:
        rec["anchor_failure"] = kind
        new_parsed, new_resolved, changed, why = try_revise_with_answer_check(
            sample, resp, dom, parsed, resolved, None, f"anchor_{kind}", attempt_log=rec.setdefault("revision_attempts", []))
        rec["revise_history"].append({"trigger": f"anchor_{kind}", "outcome": why, "changed": changed})
        if changed:
            rec["revised_flag"] = True
            rec["parsed"] = new_parsed; rec["resolved"] = new_resolved
            rec["kept_preview"] = [resp[a:b][:400] for a, b in new_resolved["kept_spans"]]
            parsed = new_parsed; resolved = new_resolved

    # ---------------- Step 4: pass 2 audit + optional revise ----------------
    if resolved["mode"] in ("spans_ok", "spans_partial", "fallback_last_to_end",
                            "fallback_tail_50pct", "whole_clean"):
        try:
            kept_text = reconstruct(resp, resolved["kept_spans"]) if resolved["kept_spans"] else ""
            rec["audit_input_spans"] = [list(s) for s in resolved["kept_spans"]]
            audit = coherence_audit(sample, kept_text)
            verified, rejected = verify_missed_masks(resp, resolved["kept_spans"],
                                                      audit.get("missed_masks", []))
            audit["missed_masks_verified"] = verified
            audit["missed_masks_rejected"] = [{"reason": rr, "quote": q} for rr, q in rejected]
            audit["severity_verified"] = verified_severity(
                audit.get("issues"), verified, audit.get("derivation_shown"))
            rec["coherence_pass1"] = {k: audit[k] for k in
                                       ("coherent", "reaches_correct_answer", "derivation_shown",
                                        "severity", "severity_verified", "issues", "missed_masks",
                                        "missed_masks_verified", "missed_masks_rejected", "checked_claims", "conflict_pairs")}
            if looks_missing_derivation(resp, resolved["kept_spans"]):
                rec["long_resp_over_mask"] = True
                if audit["severity_verified"] != "major":
                    audit["severity_verified"] = "major"

            if audit["severity_verified"] == "major":
                new_parsed, new_resolved, changed, why = try_revise_with_answer_check(
                    sample, resp, dom, parsed, resolved, audit, "coherence_major", attempt_log=rec.setdefault("revision_attempts", []))
                rec["revise_history"].append({"trigger": "coherence_major", "outcome": why, "changed": changed})
                if changed:
                    rec["revised_flag"] = True
                    rec["parsed"] = new_parsed; rec["resolved"] = new_resolved
                    rec["kept_preview"] = [resp[a:b][:400] for a, b in new_resolved["kept_spans"]]
                    parsed = new_parsed; resolved = new_resolved
                    # I1b: re-check anchor failure on new resolved
                    kind2 = anchor_failure_kind(resolved, parsed)
                    if kind2 is not None:
                        rec["post_revise_anchor_failure"] = kind2
        except Exception as pass2_exc:
            rec["pass2_error"] = repr(pass2_exc)

    # ---------------- Step 5: WCS post-hoc (whole_clean sanity) ----------------
    if resolved["mode"] == "whole_clean":
        hits = whole_clean_suspect(resp)
        rec["whole_clean_suspect"] = hits or None
        if hits and not rec["revised_flag"]:
            fake_audit = {"issues": [f"heuristic: {list(hits.keys())}"],
                          "missed_masks_verified": [f"hit: {k}" for k in hits]}
            new_parsed, new_resolved, changed, why = try_revise_with_answer_check(
                sample, resp, dom, parsed, resolved, fake_audit, "whole_clean_suspect", attempt_log=rec.setdefault("revision_attempts", []))
            rec["revise_history"].append({"trigger": "wcs", "outcome": why, "changed": changed})
            if changed:
                rec["revised_flag"] = True
                rec["parsed"] = new_parsed; rec["resolved"] = new_resolved
                rec["kept_preview"] = [resp[a:b][:400] for a, b in new_resolved["kept_spans"]]
                parsed = new_parsed; resolved = new_resolved
                # I1b: re-check anchor failure
                kind3 = anchor_failure_kind(resolved, parsed)
                if kind3 is not None:
                    rec["post_wcs_anchor_failure"] = kind3
                    # try one more anchor revise
                    ap, ar, ac, aw = try_revise_with_answer_check(
                        sample, resp, dom, parsed, resolved, None, f"anchor_{kind3}", attempt_log=rec.setdefault("revision_attempts", []))
                    rec["revise_history"].append({"trigger": f"post_wcs_anchor_{kind3}",
                                                   "outcome": aw, "changed": ac})
                    if ac:
                        rec["parsed"] = ap; rec["resolved"] = ar
                        rec["kept_preview"] = [resp[a:b][:400] for a, b in ar["kept_spans"]]
                        parsed = ap; resolved = ar

    # ---------------- Step 6: F4 duplicate answer trim ----------------
    if resolved and resolved.get("kept_spans"):
        trimmed_spans, was_trimmed = trim_duplicate_answer(resp, resolved["kept_spans"])
        if was_trimmed:
            L = len(resp)
            new_frac = sum(b - a for a, b in trimmed_spans) / L
            rec["resolved"]["kept_spans"] = trimmed_spans
            rec["resolved"]["kept_frac"] = new_frac
            rec["duplicate_answer_trimmed"] = True
            rec["kept_preview"] = [resp[a:b][:400] for a, b in trimmed_spans]

    # ---------------- Step 7: I3 — thin kept text → dropped ----------------
    if rec.get("resolved") and rec["resolved"].get("mode") not in ("dropped",):
        kept_spans = rec["resolved"].get("kept_spans") or []
        if is_thin_kept_text(resp, kept_spans, dom):
            # override to dropped
            rec["thin_kept_text_dropped"] = {
                "prev_mode": rec["resolved"]["mode"],
                "prev_kept_frac": rec["resolved"]["kept_frac"],
                "prev_kept_spans": kept_spans,
            }
            rec["resolved"] = {"mode": "dropped", "kept_spans": [], "kept_frac": 0.0,
                                "spans_status": []}
            rec["kept_preview"] = []

    return rec



# Final acceptance gate: at most two additional repair calls. No guessed masks.
def finalize_revision_gate(sample, dom, rec, max_repairs=2):
    resp = sample["response"]
    gate = {"version": 1, "status": "pending", "attempts": [], "audits": [],
            "confirmed_findings": [], "errors": []}
    rec["quality_gate"] = gate
    if (rec.get("resolved") or {}).get("mode") == "dropped":
        rec.update(training_eligible=False, needs_review=False)
        gate["status"] = "dropped"
        return rec
    ledger=[]
    first_audit=rec.get("coherence_pass1")
    audit_errors=[]
    if first_audit:
        ledger, audit_errors=guard.audit_evidence(resp,
            (rec.get("resolved_pass1") or rec["resolved"]).get("kept_spans", []), first_audit)
        # Use the exact kept spans audited, which may follow an anchor repair.
        if rec.get("audit_input_spans") is not None:
            ledger,audit_errors=guard.audit_evidence(resp,rec["audit_input_spans"],first_audit)
    elif not rec.get("mode_source") == "if_shortcut":
        audit_errors=['audit_missing_or_failed']
    gate["confirmed_findings"] = ledger
    need_reaudit=bool(rec.get("revised_flag") or audit_errors or ledger or rec.get("pass2_error"))
    current_audit=first_audit or {}
    for attempt in range(max_repairs+1):
        resolved=rec.get("resolved") or {}
        parsed=rec.get("parsed")
        errors=guard.candidate_errors(resp,parsed,resolved,kept_covers_final_answer,dom)
        spans=resolved.get("kept_spans",[])
        remaining=[f for f in ledger if guard.intersects(spans,f["interval"])]
        if remaining:errors.append('confirmed_error_still_supervised')
        if not errors and resolved.get('mode')=='dropped':
            gate.update(status='dropped', errors=[])
            rec.update(training_eligible=False, needs_review=False)
            return rec
        if not errors and need_reaudit:
            try:
                current_audit=coherence_audit(sample,reconstruct(resp,spans))
                gate['audits'].append(current_audit)
                new_findings,audit_errors=guard.audit_evidence(resp,spans,current_audit)
                for f in new_findings:
                    if f not in ledger:ledger.append(f)
                errors.extend(audit_errors)
                if new_findings:errors.append('confirmed_error_still_supervised')
            except Exception as exc:
                errors.append('post_revision_audit_error:'+repr(exc))
        elif not errors:
            errors.extend(audit_errors)
        if not errors:
            gate.update(status='accepted',errors=[])
            rec.update(training_eligible=True,needs_review=False)
            return rec
        gate['errors']=errors
        if attempt==max_repairs:break
        # Do not edit quotations or subtract raw intervals automatically.
        feedback={"issues":list(current_audit.get('issues') or [])+errors,
            "missed_masks_verified":[f['quote'] for f in ledger],
            "missed_masks":[f['quote'] for f in ledger]}
        if gate['attempts']:
            feedback['last_repair_failure']=gate['attempts'][-1].get('outcome')
        prompt=(build_user(sample)+"\n\n[REVISION ACCEPTANCE FAILURE]\n"+
            json.dumps({"candidate":parsed,"feedback":feedback},ensure_ascii=False)+
            "\nReturn a corrected annotation as strictly valid JSON. Use exact unique "
            "verbatim start/end anchors, with JSON quotes and backslashes escaped. "
            "Do not change punctuation or whitespace in decoded anchors. Keep the complete "
            "answer and a self-contained correct derivation. Confirmed erroneous quotes "
            "must not overlap kept spans; do not merely claim they were masked. If no "
            "complete correct solution can be extracted, set whole_clean=false, "
            "has_clean_final=false, keep_spans=[]. Do not invent missing proof.")
        log={'attempt':attempt+1,'input_errors':list(errors)}
        gate['attempts'].append(log)
        try:
            raw=call(SYSTEM_PROMPT_V4,prompt,temperature=0.0)
            log['raw']=raw
            new_parsed,err=guard.parse_annotation(raw)
            if err:
                log['outcome']=err
                continue
            new_resolved=resolve_spans(resp,new_parsed)
            errs=guard.candidate_errors(resp,new_parsed,new_resolved,kept_covers_final_answer,dom)
            if any(guard.intersects(new_resolved.get('kept_spans',[]),f['interval']) for f in ledger):
                errs.append('confirmed_error_still_supervised')
            log.update(parsed=new_parsed,resolved=new_resolved)
            if errs:
                log['outcome']=','.join(errs)
                continue
            rec.update(parsed=new_parsed,resolved=new_resolved,revised_flag=True,
                kept_preview=[resp[a:b][:400] for a,b in new_resolved['kept_spans']])
            log['outcome']='candidate_valid_pending_audit'
            need_reaudit=True
        except Exception as exc:
            log['outcome']='repair_exception:'+repr(exc)
    # Compatible with existing exporters: unresolved cases have NO training spans.
    rec['candidate_before_quarantine']={k:rec.get(k) for k in ('parsed','resolved','kept_preview')}
    rec.update(training_eligible=False,needs_review=True,
        parsed={"whole_clean":False,"has_clean_final":False,"keep_spans":[],
                "mask_spans_summary":[],"reason":"quality gate unresolved; needs review",
                "risk_tags":["quality_gate_unresolved"]},
        resolved={"mode":"dropped","kept_spans":[],"kept_frac":0.0,"spans_status":[]},
        kept_preview=[])
    gate['status']='needs_review'
    return rec


def annotate(job, use_if_shortcut=False):
    rec=_annotate_legacy(job,use_if_shortcut=use_if_shortcut)
    if not rec.get('resolved'):
        rec.update(training_eligible=False,needs_review=True,
            quality_gate={"version":1,"status":"needs_review","errors":[rec.get('error') or 'missing_result']})
        return rec
    return finalize_revision_gate(job[3],job[0],rec)


# =============================================================================
# sampling + driver (same as v5)
# =============================================================================

def passed(sample):
    dom = sample["metadata"]["domain"]; r = sample["reward"]
    if dom == "if": return bool(r.get("follow_all_instructions"))
    return r.get("score") == 1.0


def collect_reward_one(rollout_ids):
    pool = {"math": [], "codeio": [], "if": []}
    for rid in rollout_ids:
        fp = ROLLOUT_ROOT / f"{rid}.pt"
        if not fp.exists(): continue
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
        print(f"[sample] {dom}: {len(chosen)} of {len(arr)}", flush=True)
    picks.extend(pool["if"])
    print(f"[sample] if: added {len(pool['if'])}", flush=True)
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
        if r.get("thin_kept_text_dropped"): special["thin_kept_text_dropped"] += 1
        if r.get("anchor_failure") == "standard": special["anchor_failure_standard"] += 1
        if r.get("anchor_failure") == "hidden": special["anchor_failure_hidden"] += 1
        if r.get("post_revise_anchor_failure"): special["post_revise_anchor_failure"] += 1
        if r.get("post_wcs_anchor_failure"): special["post_wcs_anchor_failure"] += 1
        if r.get("long_resp_over_mask"): special["long_resp_over_mask"] += 1
        if r.get("duplicate_answer_trimmed"): special["duplicate_answer_trimmed"] += 1
        if r.get("whole_clean_suspect"): special["whole_clean_suspect"] += 1
        # count revise revert (from history)
        for h in r.get("revise_history", []):
            if h.get("outcome") == "revise_lost_final_answer":
                special["revise_reverted_answer_lost"] += 1
                break
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
    print(f"[start] target={TARGET} + all-IF | rollouts={ROLLOUT_IDS} | seed={RNG_SEED}", flush=True)
    pool = collect_reward_one(ROLLOUT_IDS)
    picks = sample_pool(pool, TARGET, RNG_SEED)
    print(f"[plan] total picks: {len(picks)} (non-IF: {sum(1 for p in picks if p[3]['metadata']['domain']!='if')})", flush=True)
    done = load_processed(RESULTS)
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
                           "resolved": None, "error": f"outer:{exc!r}"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                res = rec.get("resolved") or {}
                el = time.time() - t0
                flags = []
                if rec.get("nothink_violation"): flags.append("NOTHINK")
                if rec.get("thin_kept_text_dropped"): flags.append("THIN")
                if rec.get("anchor_failure"): flags.append(f"ANCH-{rec['anchor_failure'][:3].upper()}")
                if rec.get("duplicate_answer_trimmed"): flags.append("DUP")
                if rec.get("whole_clean_suspect"): flags.append("WCS")
                if rec.get("long_resp_over_mask"): flags.append("LONG")
                # revert flag
                for h in rec.get("revise_history") or []:
                    if h.get("outcome") == "revise_lost_final_answer":
                        flags.append("REVERT"); break
                flag_str = "|".join(flags) if flags else "-"
                print(f"[{i}/{len(jobs)}] r{rec.get('rollout_id')}#{rec.get('sample_idx')} "
                      f"{rec['domain']}/{rec.get('prompt_hash')} mode={res.get('mode','ERR')} "
                      f"kept={(res.get('kept_frac') or 0):.2f} rev={rec.get('revised_flag')} "
                      f"flag={flag_str} [{el:.0f}s]", flush=True)
    print(f"\n[all-done] {time.time()-t0:.0f}s", flush=True)
    all_results = [json.loads(l) for l in RESULTS.open()]
    stats = summarise(all_results)
    stats["config"] = {"rollout_ids": ROLLOUT_IDS, "target": TARGET, "seed": RNG_SEED,
                       "max_workers": MAX_WORKERS, "total_samples": len(all_results), "version": "single_turn_release"}
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[stats] wrote {STATS}"); print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
