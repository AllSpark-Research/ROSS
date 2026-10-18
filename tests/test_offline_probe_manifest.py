import asyncio
import json
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sglang_router", MagicMock())

from slime.rollout import offline_probe_manifest as manifest_module
from slime.rollout import tg_opd_rollout as tg
from slime.rollout.offline_probe_manifest import OfflineProbeManifest, build_manifest, sample_fingerprint
from slime.utils.types import Sample


def _sample(index=0, reward=0.0, *, prompt="p", metadata=None):
    return Sample(
        group_index=0,
        index=index,
        prompt=prompt,
        label="label",
        metadata=metadata or {"domain": "math"},
        tokens=[1, 2],
        response="x",
        response_length=1,
        reward=reward,
        status=Sample.Status.COMPLETED,
    )


def _write_manifest(path, sample, *, reliable, pass_count, threshold=1.0):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {"probe_n": 10, "min_pass": 7, "reward_threshold": threshold},
                "records": {
                    sample_fingerprint(sample): {
                        "scores": [1.0] * pass_count + [0.0] * (10 - pass_count),
                        "pass_count": pass_count,
                        "reliable": reliable,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_fingerprint_ignores_runtime_indices_and_dict_order():
    left = _sample(index=1, metadata={"b": 2, "a": 1})
    left.group_index = 10
    right = _sample(index=999, metadata={"a": 1, "b": 2})
    right.group_index = 999
    assert sample_fingerprint(left) == sample_fingerprint(right)


@pytest.mark.unit
def test_manifest_requires_exact_gate_settings(tmp_path):
    sample = _sample()
    path = tmp_path / "manifest.json"
    _write_manifest(path, sample, reliable=True, pass_count=7)
    with pytest.raises(ValueError, match="settings mismatch"):
        OfflineProbeManifest(str(path), probe_n=10, min_pass=8, reward_threshold=1.0)


@pytest.mark.unit
def test_build_manifest_keeps_scores_and_applies_seven_of_ten_boundary(tmp_path, monkeypatch):
    rollout = tmp_path / "rollouts"
    rollout.mkdir()
    dump = rollout / "rollout_0.pt"
    dump.touch()
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    samples = []
    for group_index, passes in ((0, 7), (1, 6)):
        for index in range(10):
            sample = _sample(index=index, reward={"score": 1.0 if index < passes else 0.0}, prompt=f"p{group_index}")
            sample.group_index = group_index
            samples.append(sample.to_dict())
    monkeypatch.setattr(manifest_module, "_load_rollout_samples", lambda _path: samples)

    output = tmp_path / "manifest.json"
    built = build_manifest(
        rollout_dir=str(rollout),
        output_path=str(output),
        probe_n=10,
        min_pass=7,
        reward_threshold=1.0,
        reward_key="score",
        dataset_path=str(dataset),
        teacher_checkpoint="teacher",
        probe_max_response_len=8192,
    )

    records = list(built["records"].values())
    assert [record["pass_count"] for record in records] == [7, 6]
    assert [record["reliable"] for record in records] == [True, False]
    assert all(len(record["scores"]) == 10 for record in records)


def _rollout_args():
    return Namespace(
        reward_key=None,
        tg_opd_mode="offline_tg_opd",
        tg_opd_select_k=4,
        tg_opd_select_strategy="high_low",
        tg_opd_unreliable_action="grpo",
        n_samples_per_prompt=8,
    )


def _install_student(monkeypatch):
    async def generate(_args, _group, sampling_params, evaluation):
        return [_sample(index=i, reward=reward) for i, reward in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7])]

    monkeypatch.setattr(tg, "generate_and_rm_group", generate)
    monkeypatch.setattr(
        tg.sglang_rollout,
        "call_dynamic_filter",
        lambda _dynamic_filter, _args, _group: SimpleNamespace(keep=True, reason=None),
    )


def _teacher_state(reliable, pass_count):
    state = MagicMock()
    state.domain_for.return_value = "math"
    state.teacher_for.return_value = "teacher"
    state.offline_manifest.lookup.return_value = {"reliable": reliable, "pass_count": pass_count}
    return state


@pytest.mark.unit
def test_offline_fallback_skips_probe_and_teacher_logprob(monkeypatch):
    _install_student(monkeypatch)
    monkeypatch.setattr(tg, "_teacher_probe", MagicMock(side_effect=AssertionError("online probe called")))
    teacher_logprob = MagicMock(side_effect=AssertionError("teacher logprob called"))
    monkeypatch.setattr(tg, "_teacher_logprob", teacher_logprob)
    student_state = SimpleNamespace(sampling_params={"temperature": 1.0})

    selected, metrics, timing, keep, _reason, probe_record = asyncio.run(
        tg._run_one_prompt(_rollout_args(), student_state, _teacher_state(False, 6), [_sample()], None)
    )

    assert keep and probe_record is None
    assert all(sample.opd_mask == 0 for sample in selected)
    assert metrics["tg_opd/fallback_groups"] == 1.0
    assert metrics["tg_opd/teacher_logprob_skipped_groups"] == 1.0
    assert timing["logprob_wall"] == 0.0


@pytest.mark.unit
def test_offline_reliable_group_gets_teacher_logprob(monkeypatch):
    _install_student(monkeypatch)
    monkeypatch.setattr(tg, "_teacher_probe", MagicMock(side_effect=AssertionError("online probe called")))
    calls = []

    async def teacher_logprob(_args, _state, sample, _teacher_model_name):
        calls.append(sample.index)
        sample.teacher_log_probs = [0.5]

    monkeypatch.setattr(tg, "_teacher_logprob", teacher_logprob)
    student_state = SimpleNamespace(sampling_params={"temperature": 1.0})

    selected, metrics, _timing, keep, _reason, probe_record = asyncio.run(
        tg._run_one_prompt(_rollout_args(), student_state, _teacher_state(True, 7), [_sample()], None)
    )

    assert keep and probe_record is None
    assert all(sample.opd_mask == 1 for sample in selected)
    assert sorted(calls) == sorted(sample.index for sample in selected)
    assert metrics["tg_opd/opd_groups"] == 1.0
