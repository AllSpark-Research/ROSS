"""Local Math, CodeIO and instruction-following reward dispatch."""
import json
from . import math_dapo, codeio, codeio_input_exec, if_reward
from .length_reward import apply_length_reward


def default_compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, dict):
                ground_truth = parsed
        except json.JSONDecodeError:
            pass
    extra_info = dict(extra_info or {})
    if data_source == "math_dapo" or str(data_source).startswith("aime"):
        return math_dapo.compute_score(solution_str, ground_truth)
    if data_source == "simulation__codeio_input_exec":
        return codeio_input_exec.compute_score(data_source, solution_str, ground_truth, extra_info)
    if data_source == "simulation__codeio":
        return codeio.compute_score(data_source, solution_str, ground_truth, extra_info)
    if data_source in {"nemotron_if_rl", "ifbench", "ifeval", "instruction_following"}:
        result = if_reward.compute_score(data_source, solution_str, ground_truth, extra_info)
        return apply_length_reward(result, solution_str, extra_info)
    raise ValueError(f"Unsupported released data source: {data_source!r}")
