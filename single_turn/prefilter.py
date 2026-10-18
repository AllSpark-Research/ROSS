#!/usr/bin/env python3
"""Mechanical prefilter for verifier-positive single-turn rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$")
PY_BLOCK_RE = re.compile(r"```\s*python\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
ACTION_TAG_RE = re.compile(
    r"^\[(ASSESS|ADVANCE|VERIFY|SIMPLIFY|SYNTHESIZE|PIVOT|OUTPUT)\]",
    re.MULTILINE,
)

# Deliberately phrase-based. Bare "wrong" or "contradiction" can be legitimate
# in problem statements or proof-by-contradiction solutions.
STRONG_ERROR_RE = re.compile(
    r"\b("
    r"this\s+(?:is|was)\s+wrong|"
    r"(?:it|this|that|my\s+previous\s+(?:approach|solution|answer|reasoning))\s+"
    r"(?:is|was)\s+(?:incorrect|wrong|not\s+correct)|"
    r"not\s+correct\s+(?:because|since|as)|"
    r"there\s+(?:is|was)\s+(?:a\s+)?mistake|"
    r"I\s+(?:made|have\s+made)\s+(?:a\s+)?mistake|"
    r"my\s+previous|"
    r"let'?s\s+correct|"
    r"we\s+need\s+to\s+correct|"
    r"where\s+is\s+the\s+error|"
    r"the\s+error\s+is|"
    r"re-?evaluate\s+carefully|"
    r"restart\s+the\s+(?:solution|approach|analysis)|"
    r"start\s+over"
    r")\b",
    re.IGNORECASE,
)


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_json(line: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(line)
    except Exception as exc:  # noqa: BLE001
        return None, f"bad_json:{type(exc).__name__}"
    if not isinstance(obj, dict):
        return None, "not_object"
    return obj, None


def normalize_messages(obj: dict[str, Any]) -> tuple[list[dict[str, str]] | None, str | None]:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return None, "missing_messages"

    user = None
    assistant = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if role == "user" and user is None:
            user = content
        elif role == "assistant":
            assistant = content

    if not user or not assistant:
        return None, "missing_user_or_assistant"

    # Use a uniform two-message SFT format to avoid mixing system prompts.
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], None


def reject_if_common_bad(assistant: str, min_len: int, max_len: int) -> str | None:
    if len(assistant) < min_len:
        return "assistant_too_short"
    if len(assistant) > max_len:
        return "assistant_too_long"
    if STRONG_ERROR_RE.search(assistant):
        return "strong_error_or_self_correction"
    return None


def math_reject_reason(assistant: str, args: argparse.Namespace) -> str | None:
    reason = reject_if_common_bad(assistant, args.min_assistant_chars, args.max_assistant_chars)
    if reason:
        return reason

    answers = list(ANSWER_RE.finditer(assistant))
    if len(answers) == 0:
        return "math_missing_answer_line"
    if len(answers) > 1:
        return "math_multiple_answer_lines"

    if answers[-1].start() < len(assistant) * args.answer_min_position_ratio:
        return "math_answer_not_near_end"

    if args.reject_math_code_fence and "```" in assistant:
        return "math_contains_code_fence"

    return None


def code_reject_reason(assistant: str, args: argparse.Namespace) -> str | None:
    reason = reject_if_common_bad(assistant, args.min_assistant_chars, args.max_assistant_chars)
    if reason:
        return reason

    if ACTION_TAG_RE.search(assistant):
        return "code_contains_action_tags"

    if assistant.count("```") % 2 != 0:
        return "code_unbalanced_fences"

    blocks = PY_BLOCK_RE.findall(assistant)
    if not blocks:
        return "code_missing_python_block"

    code = blocks[-1]
    if len(code) < args.min_code_chars:
        return "code_too_short"
    if len(code) > args.max_code_chars:
        return "code_too_long"
    if args.require_stdin and ("input(" not in code and "sys.stdin" not in code):
        return "code_no_stdin_read"

    return None


def write_jsonl_line(handle: Any, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def process_file(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_path = Path(args.rejected)
    stats_path = Path(args.stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    kept_prompt_hashes: set[str] = set()
    seen_pair_hashes: set[str] = set()

    reject_fn = math_reject_reason if args.domain == "math" else code_reject_reason

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout, rejected_path.open("w", encoding="utf-8") as frej:
        for line_no, line in enumerate(fin, start=1):
            counters["rows_total"] += 1
            obj, err = load_json(line)
            if err:
                counters[f"reject:{err}"] += 1
                write_jsonl_line(frej, {"line_no": line_no, "reason": err, "raw": line.rstrip("\n")})
                continue

            messages, err = normalize_messages(obj or {})
            if err:
                counters[f"reject:{err}"] += 1
                write_jsonl_line(frej, {"line_no": line_no, "reason": err, "record": obj})
                continue

            user = messages[0]["content"]
            assistant = messages[1]["content"]
            reason = reject_fn(assistant, args)
            if reason:
                counters[f"reject:{reason}"] += 1
                write_jsonl_line(
                    frej,
                    {
                        "line_no": line_no,
                        "reason": reason,
                        "prompt_hash": md5_text(user),
                        "assistant_hash": md5_text(assistant),
                    },
                )
                continue

            pair_hash = md5_text(user + "\n<assistant>\n" + assistant)
            if args.drop_exact_duplicates and pair_hash in seen_pair_hashes:
                counters["reject:duplicate_pair"] += 1
                continue

            seen_pair_hashes.add(pair_hash)
            kept_prompt_hashes.add(md5_text(user))
            counters["rows_kept"] += 1
            write_jsonl_line(fout, {"messages": messages})

    stats = {
        "domain": args.domain,
        "input": str(input_path),
        "output": str(output_path),
        "rejected": str(rejected_path),
        "unique_prompts_kept": len(kept_prompt_hashes),
        "counters": dict(counters),
        "config": {
            "min_assistant_chars": args.min_assistant_chars,
            "max_assistant_chars": args.max_assistant_chars,
            "answer_min_position_ratio": args.answer_min_position_ratio,
            "reject_math_code_fence": args.reject_math_code_fence,
            "min_code_chars": args.min_code_chars,
            "max_code_chars": args.max_code_chars,
            "require_stdin": args.require_stdin,
            "drop_exact_duplicates": args.drop_exact_duplicates,
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["math", "code"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--min-assistant-chars", type=int, default=500)
    parser.add_argument("--max-assistant-chars", type=int, default=26000)
    parser.add_argument("--answer-min-position-ratio", type=float, default=0.80)
    parser.add_argument("--reject-math-code-fence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-code-chars", type=int, default=200)
    parser.add_argument("--max-code-chars", type=int, default=12000)
    parser.add_argument("--require-stdin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-exact-duplicates", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    process_file(build_parser().parse_args())


if __name__ == "__main__":
    main()
