#!/usr/bin/env python3

import argparse
import json

from slime.rollout.offline_probe_manifest import OfflineProbeManifest
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a manifest against the exact training dataset view.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--probe-n", type=int, default=10)
    parser.add_argument("--min-pass", type=int, default=7)
    parser.add_argument("--reward-threshold", type=float, required=True)
    parser.add_argument("--input-key", default="input")
    parser.add_argument("--label-key")
    parser.add_argument("--metadata-key", default="metadata")
    parser.add_argument("--tool-key")
    parser.add_argument("--max-prompt-len", type=int, required=True)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--apply-chat-template-kwargs", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chat_template_kwargs = json.loads(args.apply_chat_template_kwargs)
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
    dataset = Dataset(
        args.prompt_data,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_prompt_len,
        prompt_key=args.input_key,
        label_key=args.label_key,
        metadata_key=args.metadata_key,
        tool_key=args.tool_key,
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=chat_template_kwargs,
    )
    manifest = OfflineProbeManifest(
        args.manifest,
        probe_n=args.probe_n,
        min_pass=args.min_pass,
        reward_threshold=args.reward_threshold,
    )
    coverage = manifest.validate_samples(dataset.origin_samples)
    if coverage["missing"]:
        raise RuntimeError(f"Offline manifest coverage failed: {coverage}")

    reliable = sum(manifest.lookup(sample)["reliable"] for sample in dataset.origin_samples)
    result = {
        **coverage,
        "manifest_records": len(manifest.records),
        "reliable": reliable,
        "fallback": coverage["total"] - reliable,
        "reliable_rate": reliable / max(1, coverage["total"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
