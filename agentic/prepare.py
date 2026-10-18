#!/usr/bin/env python3
"""Validate step exports and build deterministic annotation windows."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "preprocessing"))
import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing verified *.steps.json exports")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-owned-chars", type=int, default=18000)
    args = parser.parse_args()
    pipeline.prepare(args.input, args.output, args.max_steps, args.max_owned_chars)


if __name__ == "__main__":
    main()
