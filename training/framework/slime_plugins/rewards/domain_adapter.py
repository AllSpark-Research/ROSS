from __future__ import annotations

import asyncio
import importlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

ROSS_REWARD_ROOT_ENV = "ROSS_REWARDS_ROOT"
ROSS_REWARD_MAX_CONCURRENCY_ENV = "ROSS_REWARD_MAX_CONCURRENCY"
ROSS_IF_AGGREGATION_OVERRIDE_ENV = "ROSS_IF_AGGREGATION_OVERRIDE"
_VALID_IF_AGGREGATIONS = {"strict", "instruction_avg", "avg_plus_all_bonus"}
ROSS_REWARD_MODULE = "reward_score"
_TRAILING_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>")
_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_reward_root() -> Path:
    root = os.environ.get(ROSS_REWARD_ROOT_ENV)
    if not root:
        raise RuntimeError(f"{ROSS_REWARD_ROOT_ENV} is not set.")

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise RuntimeError(f"{ROSS_REWARD_ROOT_ENV} must point to a directory, got: {root_path}")
    return root_path


def _load_reward_module():
    if not os.environ.get(ROSS_REWARD_ROOT_ENV):
        return importlib.import_module("slime_plugins.rewards.domain_verifiers")
    root_path = _resolve_reward_root()
    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        importlib.invalidate_caches()
    except KeyError:
        pass

    module = importlib.import_module(ROSS_REWARD_MODULE)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"Imported {ROSS_REWARD_MODULE!r} without a file path.")

    module_path = Path(module_file).resolve()
    if not _is_relative_to(module_path, root_path):
        raise RuntimeError(
            f"Imported {ROSS_REWARD_MODULE!r} from {module_path}, outside configured root {root_path}."
        )
    return module


def _get_reward_spec(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward_spec = metadata.get("reward_spec")
    if not isinstance(reward_spec, dict):
        raise ValueError("Domain reward requires sample.metadata['reward_spec'] to be a dict.")
    return reward_spec


def _strip_trailing_special_tokens(text: str) -> str:
    while True:
        stripped = text.rstrip()
        for token in _TRAILING_SPECIAL_TOKENS:
            if stripped.endswith(token):
                text = stripped[: -len(token)]
                break
        else:
            return stripped


def _build_solution_str(sample: Sample, reward_spec: dict[str, Any]) -> str:
    solution_mode = reward_spec.get("solution_mode", "response")
    if solution_mode == "response":
        return _strip_trailing_special_tokens(sample.response)
    if solution_mode == "prompt_plus_response":
        return _strip_trailing_special_tokens(str(sample.prompt) + sample.response)
    raise ValueError(f"Unsupported reward_spec.solution_mode: {solution_mode!r}")


def _normalize_reward_output(sample: Sample, reward: Any) -> dict[str, Any]:
    if isinstance(reward, dict):
        if "score" not in reward:
            raise ValueError(f"Domain reward returned a dict without 'score' for sample index={sample.index}.")
        return reward
    if isinstance(reward, (int, float, bool)):
        score = float(reward)
        return {"score": score, "acc": score}
    raise TypeError(f"Unsupported reward type {type(reward).__name__} for sample index={sample.index}.")


def _compute_reward(sample: Sample) -> dict[str, Any]:
    reward_module = _load_reward_module()
    reward_spec = _get_reward_spec(sample)
    data_source = reward_spec.get("data_source")
    if not data_source:
        raise ValueError("Domain reward requires reward_spec['data_source'].")

    extra_info = reward_spec.get("extra_info", sample.metadata)
    if isinstance(extra_info, dict):
        extra_info = dict(extra_info)
        extra_info.setdefault("response_length", sample.response_length)
        extra_info["overlong"] = bool(extra_info.get("overlong")) or sample.status == Sample.Status.TRUNCATED
        aggregation_override = os.environ.get(ROSS_IF_AGGREGATION_OVERRIDE_ENV)
        if aggregation_override and data_source == "nemotron_if_rl":
            if aggregation_override not in _VALID_IF_AGGREGATIONS:
                raise ValueError(
                    f"{ROSS_IF_AGGREGATION_OVERRIDE_ENV} must be one of "
                    f"{sorted(_VALID_IF_AGGREGATIONS)}, got {aggregation_override!r}"
                )
            extra_info["aggregation"] = aggregation_override

    reward = reward_module.default_compute_score(
        data_source=data_source,
        solution_str=_build_solution_str(sample, reward_spec),
        ground_truth=reward_spec.get("ground_truth", sample.label),
        extra_info=extra_info,
    )
    return _normalize_reward_output(sample, reward)


def _resolve_max_concurrency() -> int:
    raw = os.environ.get(ROSS_REWARD_MAX_CONCURRENCY_ENV, "1")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{ROSS_REWARD_MAX_CONCURRENCY_ENV} must be positive, got {value}")
    return value


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    with _EXECUTORS_LOCK:
        executor = _EXECUTORS.get(max_workers)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"domain-rm-{max_workers}")
            _EXECUTORS[max_workers] = executor
        return executor


async def _compute_reward_async(sample: Sample) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_executor(_resolve_max_concurrency()), _compute_reward, sample)


async def custom_rm(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return await asyncio.gather(*[_compute_reward_async(sample) for sample in sample_or_samples])
    return await _compute_reward_async(sample_or_samples)
