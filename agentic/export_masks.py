#!/usr/bin/env python3
"""Attach accepted character exclusions to agentic message records.

The input JSONL must contain ``sample_id`` and ``messages``. Assistant messages
must appear in the same order as the G001, G002, ... steps in the corresponding
prepared state. The exporter rejects incomplete or unresolved annotations.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = withheld = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = row["sample_id"]
            state = read(args.prepared / sample / "state.json")
            ann = read(args.annotations / "final" / sample / "annotation.json")
            coverage = ann.get("coverage", {})
            unresolved = ann.get("semantic_uncertain", []) + ann.get("localization_unresolved", [])
            if not coverage.get("complete") or unresolved:
                withheld += 1
                continue
            step_ids = [step["id"] for step in state["steps"]]
            assistant = [message for message in row["messages"] if message.get("role") == "assistant"]
            if len(assistant) != len(step_ids):
                raise ValueError(f"{sample}: {len(assistant)} assistant messages for {len(step_ids)} steps")
            by_step = defaultdict(list)
            for span in ann.get("mask_spans", []):
                by_step[span["region_id"]].append([span["char_start"], span["char_end"]])
            for step_id, message in zip(step_ids, assistant):
                spans = sorted(by_step[step_id])
                if any(a < 0 or a >= b or (i and a < spans[i - 1][1])
                       for i, (a, b) in enumerate(spans)):
                    raise ValueError(f"{sample}/{step_id}: invalid or overlapping exclusions")
                message["sft_region_exclude"] = spans
            target.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"exported": kept, "withheld": withheld}))


if __name__ == "__main__":
    main()
