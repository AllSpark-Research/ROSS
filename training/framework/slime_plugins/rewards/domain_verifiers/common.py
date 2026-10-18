from __future__ import annotations

import json
import re
from typing import Any

TRAILING_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>")


def strip_trailing_special_tokens(text: str) -> str:
    text = str(text or "")
    while True:
        stripped = text.rstrip()
        for token in TRAILING_SPECIAL_TOKENS:
            if stripped.endswith(token):
                text = stripped[: -len(token)]
                break
        else:
            return stripped


def maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def coerce_dict(value: Any) -> dict[str, Any]:
    value = maybe_json_loads(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def coerce_list(value: Any) -> list[Any]:
    value = maybe_json_loads(value)
    if isinstance(value, list):
        return list(value)
    return []


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_tagged_block(text: str, tag: str) -> str | None:
    pattern = rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>"
    match = re.search(pattern, ensure_text(text), re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def strip_json_code_fence(text: str) -> str:
    raw = ensure_text(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return raw


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
