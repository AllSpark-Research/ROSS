from __future__ import annotations

import logging
from collections import defaultdict
from numbers import Real
from typing import Any

logger = logging.getLogger(__name__)


def _extract_score(reward: Any) -> float | None:
    if isinstance(reward, dict):
        score = reward.get("score")
        if isinstance(score, Real):
            return float(score)
        return None

    if isinstance(reward, Real):
        return float(reward)

    return None


def _get_rollout_step(args, rollout_id: int) -> int:
    from slime.ray.rollout import compute_rollout_step

    return compute_rollout_step(args, rollout_id)


def _log_metrics(args, log_dict: dict[str, float]) -> None:
    from slime.utils import logging_utils

    logging_utils.log(args, log_dict, step_key="rollout/step")


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    scores_by_data_source = defaultdict(list)
    gate_open_values = []
    teacher_pass_rates = []
    teacher_verify_latencies = []
    teacher_verify_timeouts = []
    teacher_verify_timeout_opd_fallbacks = []
    teacher_verify_gate_would_open_values = []
    teacher_verify_nonblockings = []
    teacher_verify_cancelled = []

    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        reward_spec = metadata.get("reward_spec") if isinstance(metadata.get("reward_spec"), dict) else {}
        data_source = reward_spec.get("data_source")

        if "mopd_gate_open" in metadata:
            gate_open_values.append(1.0 if metadata.get("mopd_gate_open") else 0.0)
        if metadata.get("mopd_teacher_pass_rate") is not None:
            teacher_pass_rates.append(float(metadata.get("mopd_teacher_pass_rate") or 0.0))
        if metadata.get("mopd_teacher_verify_latency_s") is not None:
            teacher_verify_latencies.append(float(metadata.get("mopd_teacher_verify_latency_s") or 0.0))
        if "mopd_teacher_verify_timeout" in metadata:
            teacher_verify_timeouts.append(1.0 if metadata.get("mopd_teacher_verify_timeout") else 0.0)
        if "mopd_teacher_verify_timeout_fallback_used" in metadata:
            teacher_verify_timeout_opd_fallbacks.append(
                1.0 if metadata.get("mopd_teacher_verify_timeout_fallback_used") else 0.0
            )
        if "mopd_teacher_verify_gate_would_open" in metadata:
            teacher_verify_gate_would_open_values.append(
                1.0 if metadata.get("mopd_teacher_verify_gate_would_open") else 0.0
            )
        if "mopd_teacher_verify_nonblocking" in metadata:
            teacher_verify_nonblockings.append(1.0 if metadata.get("mopd_teacher_verify_nonblocking") else 0.0)
        if "mopd_teacher_verify_cancelled" in metadata:
            teacher_verify_cancelled.append(1.0 if metadata.get("mopd_teacher_verify_cancelled") else 0.0)

        if not data_source:
            continue

        score = _extract_score(getattr(sample, "reward", None))
        if score is None:
            continue

        scores_by_data_source[str(data_source)].append(score)

    if not scores_by_data_source and not gate_open_values:
        return False

    log_dict = {
        f"rollout/data_source_score/{data_source}": sum(scores) / len(scores)
        for data_source, scores in scores_by_data_source.items()
    }
    if gate_open_values:
        log_dict["mopd/gate_open_rate"] = sum(gate_open_values) / len(gate_open_values)
    if teacher_pass_rates:
        log_dict["mopd/teacher_pass_rate_mean"] = sum(teacher_pass_rates) / len(teacher_pass_rates)
    if teacher_verify_latencies:
        log_dict["mopd/teacher_verify_latency_mean"] = sum(teacher_verify_latencies) / len(teacher_verify_latencies)
    if teacher_verify_timeouts:
        log_dict["mopd/teacher_verify_timeout_rate"] = sum(teacher_verify_timeouts) / len(teacher_verify_timeouts)
    if teacher_verify_timeout_opd_fallbacks:
        log_dict["mopd/teacher_verify_timeout_opd_fallback_rate"] = (
            sum(teacher_verify_timeout_opd_fallbacks) / len(teacher_verify_timeout_opd_fallbacks)
        )
    if teacher_verify_gate_would_open_values:
        log_dict["mopd/teacher_verify_gate_would_open_rate"] = (
            sum(teacher_verify_gate_would_open_values) / len(teacher_verify_gate_would_open_values)
        )
    if teacher_verify_nonblockings:
        log_dict["mopd/teacher_verify_nonblocking_rate"] = (
            sum(teacher_verify_nonblockings) / len(teacher_verify_nonblockings)
        )
    if teacher_verify_cancelled:
        log_dict["mopd/teacher_verify_cancelled_rate"] = sum(teacher_verify_cancelled) / len(teacher_verify_cancelled)
    log_dict["rollout/step"] = _get_rollout_step(args, rollout_id)

    logger.info(f"domain rollout {rollout_id}: {log_dict}")
    _log_metrics(args, log_dict)
    return False
