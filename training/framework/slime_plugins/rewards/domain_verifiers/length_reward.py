from __future__ import annotations

from typing import Any

from .common import ensure_text, maybe_json_loads


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part])


def _resolve_length_config(extra_info: dict[str, Any]) -> dict[str, Any] | None:
    raw_config = maybe_json_loads(extra_info.get("global_length_reward"))
    if raw_config in (None, False):
        return None
    if raw_config is True:
        raw_config = {}
    if not isinstance(raw_config, dict):
        return None
    if raw_config.get("enabled", True) is False:
        return None

    return {
        "unit": str(raw_config.get("unit", "tokens")).strip().lower() or "tokens",
        "min_length": _coerce_positive_int(raw_config.get("min_length"), 32),
        "max_length": _coerce_positive_int(raw_config.get("max_length"), 4096),
    }


def _resolve_length_value(solution_str: Any, extra_info: dict[str, Any], unit: str) -> tuple[int, str]:
    if unit in {"tokens", "token"}:
        token_length = extra_info.get("response_length", extra_info.get("token_length"))
        try:
            if token_length is not None:
                return max(int(token_length), 0), "tokens"
        except (TypeError, ValueError):
            pass
        return len(ensure_text(solution_str)), "chars"

    if unit in {"words", "word"}:
        return _word_count(ensure_text(solution_str)), "words"

    return len(ensure_text(solution_str)), "chars"


def compute_length_multiplier(
    solution_str: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    extra_info = extra_info or {}
    config = _resolve_length_config(extra_info)
    if config is None:
        return None

    min_length = config["min_length"]
    max_length = max(config["max_length"], min_length)
    length_value, resolved_unit = _resolve_length_value(solution_str, extra_info, config["unit"])

    if length_value <= 0:
        multiplier = 0.0
    elif length_value < min_length:
        multiplier = length_value / float(min_length)
    elif length_value <= max_length:
        multiplier = 1.0
    else:
        multiplier = max_length / float(length_value)

    return {
        "multiplier": float(max(0.0, min(1.0, multiplier))),
        "length": int(length_value),
        "unit": resolved_unit,
        "min_length": int(min_length),
        "max_length": int(max_length),
    }


def apply_length_reward(
    reward: dict[str, Any],
    solution_str: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = compute_length_multiplier(solution_str, extra_info)
    if info is None:
        return reward

    shaped = dict(reward)
    base_score = float(shaped.get("score", 0.0))
    shaped["score_before_length_reward"] = base_score
    shaped["length_reward_multiplier"] = info["multiplier"]
    shaped["length_reward_score"] = info["multiplier"]
    shaped["length_reward_length"] = info["length"]
    shaped["length_reward_unit"] = info["unit"]
    shaped["length_reward_min_length"] = info["min_length"]
    shaped["length_reward_max_length"] = info["max_length"]
    shaped["score"] = base_score * info["multiplier"]
    return shaped
