#!/usr/bin/env python3
"""Run ROSS single-turn annotation on verifier-positive JSONL records.

Each input record must contain ``prompt``, ``response``, ``reward``, and
``metadata``. The output preserves the annotator's parsed decision, resolved
character spans, quality gate, and eligibility decision.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import threading
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOCAL = threading.local()


def load_annotator():
    spec = importlib.util.spec_from_file_location("ross_single_turn", HERE / "annotate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def headers() -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    key = os.environ.get("ROSS_API_KEY", "")
    if key:
        name = os.environ.get("ROSS_AUTH_HEADER", "Authorization")
        scheme = os.environ.get("ROSS_AUTH_SCHEME", "Bearer")
        result[name] = f"{scheme} {key}".strip()
    return result


def call_judge(system: str, user: str, max_tokens: int = 32768,
               temperature: float = 0.0) -> str:
    body = {
        "model": os.environ.get("ROSS_MODEL", "judge-model"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    endpoint = os.environ.get("ROSS_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers(),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"incomplete judge output: {choice.get('finish_reason')}")
    message = choice["message"]
    content = message.get("content") or message.get("reasoning_content") or ""
    if not content:
        raise ValueError("empty judge output")
    return content


def normalize(record: dict) -> tuple[dict, str]:
    sample = json.loads(json.dumps(record))
    domain = sample.get("metadata", {}).get("domain") or sample.get("domain")
    if domain not in {"math", "codeio", "if"}:
        raise ValueError(f"unsupported domain: {domain!r}")
    sample.setdefault("metadata", {})["domain"] = domain
    return sample, domain


def annotate_one(item: tuple[int, dict]) -> dict:
    index, record = item
    sample, domain = normalize(record)
    if not hasattr(LOCAL, "annotator"):
        LOCAL.annotator = load_annotator()
        LOCAL.annotator.call = call_judge
        LOCAL.annotator.v4.call = call_judge
    annotator = LOCAL.annotator
    prompt_hash = hashlib.md5(sample["prompt"].encode("utf-8")).hexdigest()[:8]
    result = annotator.annotate((domain, prompt_hash, index, sample), use_if_shortcut=True)
    return {"index": index, "domain": domain, "prompt_hash": prompt_hash, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(annotate_one, enumerate(records)):
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
