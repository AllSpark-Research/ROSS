#!/usr/bin/env python3
"""Join ROSS single-turn annotations with their source rollouts for masked SFT.

The source and annotation files must preserve the same record indices. Accepted
examples are emitted as message records. Partial selections add
``sft_char_spans`` to the assistant message; whole-response selections omit the
field and therefore use the trainer's ordinary full assistant loss mask.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

USER_BLOCK = re.compile(r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_messages(record: dict, response: str, prompt_field: str) -> list[dict]:
    messages = copy.deepcopy(record.get("messages"))
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        assistant = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
        if assistant:
            position = assistant[-1]
            observed = messages[position].get("content") or ""
            if observed not in {response, response + "<|im_end|>"}:
                raise ValueError("source assistant content differs from response")
            messages[position]["content"] = response
        else:
            messages.append({"role": "assistant", "content": response})
        return messages

    prompt = record.get(prompt_field)
    if not isinstance(prompt, str):
        raise ValueError(f"missing string field {prompt_field!r}")
    matches = USER_BLOCK.findall(prompt)
    if not matches:
        raise ValueError("cannot recover the user turn from the rendered prompt")
    result = []
    system = record.get("system")
    if isinstance(system, str) and system:
        result.append({"role": "system", "content": system})
    result.extend([
        {"role": "user", "content": matches[-1]},
        {"role": "assistant", "content": response},
    ])
    return result


def normalized_spans(raw_spans, response_length: int, source_length: int) -> list[list[int]]:
    spans = []
    for raw in raw_spans or []:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"invalid span: {raw!r}")
        source_start, source_end = int(raw[0]), int(raw[1])
        if not 0 <= source_start < source_end <= source_length:
            raise ValueError(f"span escapes annotated response: {raw!r}")
        start = min(source_start, response_length)
        end = min(source_end, response_length)
        if end > start:
            spans.append([start, end])
    spans.sort()
    if any(start < spans[i - 1][1] for i, (start, _) in enumerate(spans) if i):
        raise ValueError("keep spans overlap")
    return spans


def convert(source: Path, annotations: Path, output: Path, prompt_field: str,
            response_field: str, keep_source_id: bool) -> Counter:
    records = read_jsonl(source)
    decisions = read_jsonl(annotations)
    by_index = {}
    for decision in decisions:
        index = decision.get("index")
        if not isinstance(index, int) or index in by_index:
            raise ValueError(f"invalid or duplicate annotation index: {index!r}")
        by_index[index] = decision
    if set(by_index) - set(range(len(records))):
        raise ValueError("annotation index is outside the source JSONL")

    output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter(source_rows=len(records), annotation_rows=len(decisions))
    with output.open("w", encoding="utf-8") as target:
        for index, record in enumerate(records):
            decision = by_index.get(index)
            if decision is None:
                stats["missing_annotation"] += 1
                continue
            if not decision.get("training_eligible", False):
                stats["withheld"] += 1
                continue
            response = record.get(response_field)
            if not isinstance(response, str):
                raise ValueError(f"record {index}: missing string response")
            source_length = len(response)
            if response.endswith("<|im_end|>"):
                response = response[:-len("<|im_end|>")]
            expected_length = decision.get("response_len_chars")
            if expected_length is not None and expected_length not in {
                len(response), len(response) + len("<|im_end|>")
            }:
                raise ValueError(f"record {index}: response changed after annotation")

            resolved = decision.get("resolved") or {}
            mode = resolved.get("mode")
            if mode in {None, "dropped", "ERR"}:
                raise ValueError(f"record {index}: eligible annotation has mode {mode!r}")
            spans = normalized_spans(resolved.get("kept_spans"), len(response), source_length)
            if mode != "whole_clean" and not spans:
                raise ValueError(f"record {index}: accepted partial annotation has no spans")

            messages = source_messages(record, response, prompt_field)
            final_assistant = next(
                (message for message in reversed(messages) if message.get("role") == "assistant"),
                None,
            )
            if final_assistant is None:
                raise ValueError(f"record {index}: no assistant message")
            if mode == "whole_clean":
                stats["whole_response"] += 1
            else:
                final_assistant["sft_char_spans"] = spans
                stats["selective_spans"] += 1

            row = {"messages": messages}
            if "tools" in record:
                row["tools"] = copy.deepcopy(record["tools"])
            if keep_source_id:
                row["source_id"] = record.get("sample_id", index)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["exported"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Original verifier-positive JSONL passed to single_turn/run.py")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="JSONL produced by single_turn/run.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--keep-source-id", action="store_true")
    args = parser.parse_args()
    stats = convert(args.source, args.annotations, args.output, args.prompt_field,
                    args.response_field, args.keep_source_id)
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
