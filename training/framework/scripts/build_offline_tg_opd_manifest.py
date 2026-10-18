#!/usr/bin/env python3

import argparse
import json

from slime.rollout.offline_probe_manifest import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict offline TG-OPD reliability manifest.")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-n", type=int, default=10)
    parser.add_argument("--min-pass", type=int, default=7)
    parser.add_argument("--reward-threshold", type=float, required=True)
    parser.add_argument("--reward-key", default="score")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--probe-max-response-len", type=int, required=True)
    parser.add_argument(
        "--extra-metadata-json",
        default="{}",
        help="Additional JSON object recorded in the manifest metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_metadata = json.loads(args.extra_metadata_json)
    if not isinstance(extra_metadata, dict):
        raise ValueError("--extra-metadata-json must be a JSON object")
    manifest = build_manifest(
        rollout_dir=args.rollout_dir,
        output_path=args.output,
        probe_n=args.probe_n,
        min_pass=args.min_pass,
        reward_threshold=args.reward_threshold,
        reward_key=args.reward_key,
        dataset_path=args.dataset_path,
        teacher_checkpoint=args.teacher_checkpoint,
        probe_max_response_len=args.probe_max_response_len,
        extra_metadata=extra_metadata,
    )
    metadata = manifest["metadata"]
    print(
        json.dumps(
            {
                "output": args.output,
                "records": metadata["record_count"],
                "reliable": metadata["reliable_count"],
                "fallback": metadata["fallback_count"],
                "distribution": metadata["pass_count_distribution"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
