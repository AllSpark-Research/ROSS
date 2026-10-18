import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any


HARNESS = r'''
import inspect, json, sys

def _jsonable(obj):
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(x) for x in obj)
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj

_candidate = json.loads(sys.argv[1])
_entrypoint = sys.argv[2]
_fn = globals()[_entrypoint]
_params = list(inspect.signature(_fn).parameters)
if isinstance(_candidate, dict):
    try:
        _result = _fn(**_candidate)
    except TypeError:
        if len(_params) == 1:
            _result = _fn(_candidate)
        else:
            raise
else:
    if len(_params) != 1:
        raise TypeError("non-dict input is only valid for single-argument main_solution")
    _result = _fn(_candidate)
print(json.dumps(_jsonable(_result), ensure_ascii=False, allow_nan=False))
'''


def _truncate(text: Any, limit: int = 500) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _coerce_ground_truth(ground_truth: Any) -> dict[str, Any] | None:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    end = 0
    for match in re.finditer(r"\{", text):
        if match.start() < end:
            continue
        try:
            obj, consumed = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        end = match.start() + consumed
        objects.append(obj)
    return objects


def _code_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:json|JSON)?\s*([\s\S]*?)```", text)


def _parse_model_input(solution_str: Any) -> tuple[str, Any | None]:
    text = "" if solution_str is None else str(solution_str)
    candidates: list[tuple[str, Any]] = []

    whole_objects = _json_objects(text)
    if whole_objects:
        candidates.append(("json", whole_objects[-1]))

    blocks = _code_blocks(text)
    for block in reversed(blocks):
        try:
            candidates.append(("json_block", json.loads(block.strip())))
            break
        except Exception:
            block_objects = _json_objects(block)
            if block_objects:
                candidates.append(("json_block", block_objects[-1]))
                break

    for kind, obj in candidates:
        if isinstance(obj, dict) and "input" in obj:
            return kind, obj["input"]
    if candidates:
        return "missing_input", None
    return "parse_fail", None


def _normalize(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return _normalize(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return _normalize(obj.item())
        except Exception:
            pass
    if isinstance(obj, tuple):
        return [_normalize(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_normalize(x) for x in obj)
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    return obj


def _close_enough(actual: Any, expected: Any, tolerance: float) -> bool:
    actual = _normalize(actual)
    expected = _normalize(expected)
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, int) and isinstance(expected, int):
        return abs(actual - expected) <= tolerance
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= tolerance
        except (OverflowError, ValueError):
            return False
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _close_enough(a, e, tolerance) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _close_enough(actual[k], expected[k], tolerance) for k in actual
        )
    return actual == expected


def _run_reference(gt: dict[str, Any], candidate_input: Any) -> tuple[bool, Any | str]:
    reference_code = gt.get("reference_code")
    if not isinstance(reference_code, str) or not reference_code.strip():
        return False, "missing reference_code"

    code = reference_code.rstrip() + "\n\n" + HARNESS
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            path = f.name
    except Exception as exc:
        return False, f"failed to write reference code: {exc!r}"

    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    python_bin = os.environ.get("CODEIO_INPUT_EXEC_PYTHON", sys.executable)
    timeout = float(gt.get("timeout", 5))

    try:
        proc = subprocess.run(
            [
                python_bin,
                path,
                json.dumps(candidate_input, ensure_ascii=True, allow_nan=False),
                str(gt.get("entrypoint", "main_solution")),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"execution timed out after {timeout:g} seconds"
    except Exception as exc:
        return False, repr(exc)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip())
    try:
        return True, json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, proc.stdout.strip()


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> dict:
    gt = _coerce_ground_truth(ground_truth)
    if gt is None:
        return {"score": 0.0, "acc": 0.0, "pred": None, "error": "ground_truth must be a JSON object"}

    parse_kind, candidate_input = _parse_model_input(solution_str)
    if candidate_input is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "parse_kind": parse_kind,
            "error": "failed to extract input JSON",
        }

    ok, actual_or_error = _run_reference(gt, candidate_input)
    if not ok:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": candidate_input,
            "parse_kind": parse_kind,
            "error": _truncate(actual_or_error),
        }

    target_output = gt.get("target_output")
    tolerance = float(gt.get("tolerance", 1e-4))
    score = float(_close_enough(actual_or_error, target_output, tolerance))
    return {
        "score": score,
        "acc": score,
        "pred": candidate_input,
        "parse_kind": parse_kind,
        "actual_output": actual_or_error,
        "target_output": target_output,
    }
