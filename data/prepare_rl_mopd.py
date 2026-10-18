#!/usr/bin/env python3
"""Combine released domains without changing their sampling weights."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "rl_mopd")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=["math", "codeio", "if"], default=["math", "codeio", "if"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(set(args.domains)) != len(args.domains):
        parser.error("Each domain may be supplied only once")
    rows = []
    for domain in args.domains:
        with (args.root / domain / "train.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["metadata"]["domain"] != domain:
                    raise ValueError("Inconsistent domain metadata")
                rows.append(row)
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(rows)} samples")


if __name__ == "__main__":
    main()
