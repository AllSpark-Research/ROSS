import asyncio
from argparse import Namespace
from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from slime.rollout import tg_opd_rollout as tg
from slime.rollout.tg_opd_rollout import (
    _emit_timing_metrics,
    _probe_pass_count,
    _resize_student_semaphore,
    _run_one_prompt,
    _select_student_samples,
    post_process_rewards,
)
from slime.utils.types import Sample


NUM_GPUS = 0


def _args(**overrides):
    base = dict(
        reward_key=None,
        tg_opd_select_k=4,
        n_samples_per_prompt=8,
        tg_opd_select_strategy="high_low",
        tg_opd_probe_pass_threshold=1.0,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    base.update(overrides)
    return Namespace(**base)


def _sample(index: int, reward: float, *, group_index: int = 0, opd_mask: int | None = None) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        tokens=[index, index + 100],
        response="x",
        response_length=1,
        reward=reward,
        status=Sample.Status.COMPLETED,
        teacher_log_probs=[0.0],
        opd_mask=opd_mask,
    )


# ---------------------------------------------------------------------------
# Pure-function unit tests (preserved from previous suite)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_select_student_samples_high_low_keeps_low2_high2_in_original_order():
    group = [_sample(i, reward) for i, reward in enumerate([0.5, 1.0, 0.0, 0.8, 0.2, 0.9, 0.1, 0.7])]

    selected = _select_student_samples(_args(), group)

    assert [sample.index for sample in selected] == [1, 2, 5, 6]


@pytest.mark.unit
def test_select_student_samples_top_and_low_modes():
    group = [_sample(i, reward) for i, reward in enumerate([0.5, 1.0, 0.0, 0.8, 0.2, 0.9, 0.1, 0.7])]

    top = _select_student_samples(_args(tg_opd_select_strategy="top"), group)
    low = _select_student_samples(_args(tg_opd_select_strategy="low"), group)

    assert [sample.index for sample in top] == [1, 3, 5, 7]
    assert [sample.index for sample in low] == [0, 2, 4, 6]


@pytest.mark.unit
def test_probe_pass_count_uses_configured_threshold():
    probes = [_sample(0, 0.49), _sample(1, 0.5), _sample(2, 0.9), _sample(3, 1.0)]

    assert _probe_pass_count(_args(tg_opd_probe_pass_threshold=1.0), probes) == 1
    assert _probe_pass_count(_args(tg_opd_probe_pass_threshold=0.5), probes) == 3


@pytest.mark.unit
def test_post_process_rewards_zeros_reliable_opd_and_normalizes_fallback_by_group():
    samples = [
        _sample(0, 0.0, group_index=0, opd_mask=1),
        _sample(1, 1.0, group_index=0, opd_mask=1),
        _sample(2, 0.0, group_index=1, opd_mask=0),
        _sample(3, 1.0, group_index=1, opd_mask=0),
    ]

    raw_rewards, rewards = post_process_rewards(_args(), samples)

    assert raw_rewards == [0.0, 1.0, 0.0, 1.0]
    assert rewards[0:2] == [0.0, 0.0]
    assert rewards[2] == pytest.approx(-0.7071053)
    assert rewards[3] == pytest.approx(0.7071053)


@pytest.mark.unit
def test_post_process_rewards_without_normalization_keeps_fallback_raw_rewards():
    samples = [
        _sample(0, 0.0, group_index=0, opd_mask=1),
        _sample(1, 1.0, group_index=0, opd_mask=0),
    ]

    raw_rewards, rewards = post_process_rewards(_args(rewards_normalization=False), samples)

    assert raw_rewards == [0.0, 1.0]
    assert rewards == [0.0, 1.0]


@pytest.mark.unit
def test_post_process_rewards_single_fallback_sample_does_not_create_nan():
    samples = [_sample(0, 1.0, group_index=0, opd_mask=0)]

    _raw_rewards, rewards = post_process_rewards(_args(), samples)

    assert rewards == [0.0]


# ---------------------------------------------------------------------------
# Async integration tests for _run_one_prompt — cover the truth table.
# ---------------------------------------------------------------------------


def _run_one_prompt_args(**overrides):
    base = dict(
        reward_key=None,
        tg_opd_mode="tg_opd",
        tg_opd_select_k=4,
        n_samples_per_prompt=8,
        tg_opd_select_strategy="high_low",
        tg_opd_probe_n=3,
        tg_opd_probe_min_pass=2,
        tg_opd_probe_pass_threshold=1.0,
    )
    base.update(overrides)
    return Namespace(**base)


def _student_group(rewards):
    return [_sample(i, r) for i, r in enumerate(rewards)]


def _student_state_mock():
    s = MagicMock()
    s.sampling_params = {"temperature": 1.0}
    return s


async def _run_complete_prompt(args, student_state, ts, group):
    # Exercise both production stages: streaming logprob, then batch-end gate.
    ts.domain_for.return_value = None
    ts.teacher_for.return_value = "teacher"
    selected, metrics, timing, keep, reason, record = await _run_one_prompt(
        args, student_state, ts, group, None
    )
    assert keep and reason is None
    accumulated = defaultdict(float, metrics)
    await tg._finalize_deferred_teacher_gate(args, [record] if record else [], accumulated)
    if record:
        assert record["task"].done()
    return selected, accumulated, timing


def _install_student_generate(monkeypatch, scored_group_factory):
    async def fake_generate_and_rm_group(_args, group, sampling_params, evaluation):
        scored = scored_group_factory(group)
        return scored
    monkeypatch.setattr(tg, "generate_and_rm_group", fake_generate_and_rm_group)


def _install_teacher_probe(monkeypatch, behavior):
    """behavior may be: a list[Sample] (return immediately), an Exception
    (raise immediately), or the string "pending" (await forever).
    """
    async def fake_teacher_probe(_args, _ts, _prompt_sample, _teacher_model_name):
        if behavior == "pending":
            await asyncio.Event().wait()
            return []  # unreachable
        if isinstance(behavior, BaseException):
            raise behavior
        # Yield once so the probe task is genuinely awaited.
        await asyncio.sleep(0)
        return behavior
    monkeypatch.setattr(tg, "_teacher_probe", fake_teacher_probe)


def _install_teacher_logprob(monkeypatch, calls_out, *, fail_on_indices=()):
    async def fake_teacher_logprob(_args, _ts, sample, _teacher_model_name):
        calls_out.append(sample.index)
        if sample.index in fail_on_indices:
            raise RuntimeError(f"teacher_logprob down for {sample.index}")
        sample.teacher_log_probs = [0.5] * sample.response_length
    monkeypatch.setattr(tg, "_teacher_logprob", fake_teacher_logprob)


@pytest.mark.unit
def test_run_one_prompt_probe_done_pass_marks_all_selected_opd(monkeypatch):
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]  # high_low picks indices 0,1,4,5
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))
    # 2 of 3 probes pass threshold 1.0 -> reliable.
    probes = [_sample(900, 1.0), _sample(901, 1.0), _sample(902, 0.0)]
    _install_teacher_probe(monkeypatch, probes)
    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls)

    selected, metrics, _timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    assert [s.index for s in selected] == [0, 1, 4, 5]
    assert all(s.opd_mask == 1 for s in selected)
    assert sorted(logprob_calls) == [0, 1, 4, 5]
    assert metrics["tg_opd/opd_groups"] == 1.0
    assert metrics["tg_opd/fallback_groups"] == 0.0
    assert metrics["tg_opd/probe_completed"] == 1.0
    assert "tg_opd/probe_errored" not in metrics
    assert "tg_opd/probe_timeout" not in metrics


@pytest.mark.unit
def test_run_one_prompt_probe_done_fail_marks_all_selected_fallback(monkeypatch):
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))
    # Only 1 of 3 probes passes -> pass_count=1 < min_pass=2 -> unreliable.
    probes = [_sample(900, 1.0), _sample(901, 0.0), _sample(902, 0.0)]
    _install_teacher_probe(monkeypatch, probes)
    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls)

    selected, metrics, _timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    assert all(s.opd_mask == 0 for s in selected)
    assert all(s.teacher_log_probs == [0.0] * s.response_length for s in selected)
    assert sorted(logprob_calls) == [0, 1, 4, 5]  # Online gate is deferred until batch end.
    assert metrics["tg_opd/opd_groups"] == 0.0
    assert metrics["tg_opd/fallback_groups"] == 1.0
    assert metrics["tg_opd/probe_completed"] == 1.0


@pytest.mark.unit
def test_run_one_prompt_probe_pending_stays_opd_and_cancels_probe(monkeypatch):
    """Probe never finishes before student; the new semantics keep us on OPD."""
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))
    _install_teacher_probe(monkeypatch, "pending")
    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls)

    selected, metrics, _timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    assert all(s.opd_mask == 1 for s in selected)
    assert sorted(logprob_calls) == [0, 1, 4, 5]
    assert metrics["tg_opd/opd_groups"] == 1.0
    assert metrics["tg_opd/probe_timeout"] == 1.0
    assert "tg_opd/probe_completed" not in metrics


@pytest.mark.unit
def test_run_one_prompt_probe_raised_stays_opd(monkeypatch):
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))
    _install_teacher_probe(monkeypatch, RuntimeError("teacher down"))
    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls)

    selected, metrics, _timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    assert all(s.opd_mask == 1 for s in selected)
    assert sorted(logprob_calls) == [0, 1, 4, 5]
    assert metrics["tg_opd/opd_groups"] == 1.0
    assert metrics["tg_opd/probe_errored"] == 1.0
    assert "tg_opd/probe_completed" not in metrics


@pytest.mark.unit
def test_run_one_prompt_per_sample_logprob_failure_isolated(monkeypatch):
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]  # selected: 0,1,4,5
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))
    probes = [_sample(900, 1.0), _sample(901, 1.0), _sample(902, 0.0)]
    _install_teacher_probe(monkeypatch, probes)
    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls, fail_on_indices={5})

    selected, metrics, _timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    by_index = {s.index: s for s in selected}
    assert by_index[5].opd_mask == 0
    assert by_index[5].teacher_log_probs == [0.0] * by_index[5].response_length
    for idx in (0, 1, 4):
        assert by_index[idx].opd_mask == 1
        assert by_index[idx].teacher_log_probs == [0.5] * by_index[idx].response_length
    assert metrics["tg_opd/teacher_logprob_failures"] == 1.0
    assert metrics["tg_opd/opd_groups"] == 1.0  # group-level still OPD


@pytest.mark.unit
def test_run_one_prompt_selected_opd_mode_skips_probe(monkeypatch):
    args = _run_one_prompt_args(tg_opd_mode="selected_opd")
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))

    # If probe is invoked in selected_opd mode this would raise.
    async def forbidden_probe(*_a, **_k):
        raise AssertionError("_teacher_probe must not be called in selected_opd mode")
    monkeypatch.setattr(tg, "_teacher_probe", forbidden_probe)

    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls)

    selected, metrics, _timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    assert all(s.opd_mask == 1 for s in selected)
    assert sorted(logprob_calls) == [0, 1, 4, 5]
    assert metrics["tg_opd/opd_groups"] == 1.0
    for k in ("tg_opd/probe_completed", "tg_opd/probe_timeout", "tg_opd/probe_errored"):
        assert k not in metrics


# ---------------------------------------------------------------------------
# Student semaphore resize — guards against GenerateState over-permitting
# student concurrency when --sglang-config splits the rollout pool.
# ---------------------------------------------------------------------------


def _semaphore_state_mock(value: int) -> MagicMock:
    """Mimic just enough of GenerateState for _resize_student_semaphore: a
    semaphore object exposing ``_value`` (asyncio.Semaphore's internal counter).
    """
    state = MagicMock()
    state.semaphore = asyncio.Semaphore(value)
    # Strip the sentinel so a fresh state always triggers resize check.
    if hasattr(state, "_tg_opd_semaphore_sized"):
        del state._tg_opd_semaphore_sized
    return state


def _resize_args(**overrides):
    base = dict(
        tg_opd_student_model_name="default",
        sglang_server_concurrency=128,
        sglang_model_num_engines={"default": 4, "teacher": 2},
    )
    base.update(overrides)
    return Namespace(**base)


@pytest.mark.unit
def test_resize_student_semaphore_shrinks_to_student_engine_count():
    args = _resize_args()  # 4 student engines, 2 teacher engines, concurrency=128
    state = _semaphore_state_mock(value=128 * 6)  # GenerateState over-permits at 6 engines

    _resize_student_semaphore(args, state)

    assert state.semaphore._value == 128 * 4
    assert state._tg_opd_semaphore_sized is True


@pytest.mark.unit
def test_resize_student_semaphore_idempotent_on_second_call():
    args = _resize_args()
    state = _semaphore_state_mock(value=128 * 6)

    _resize_student_semaphore(args, state)
    sema_after_first = state.semaphore
    _resize_student_semaphore(args, state)

    assert state.semaphore is sema_after_first  # second call must not rebuild


@pytest.mark.unit
def test_resize_student_semaphore_noop_when_no_per_model_split_known():
    args = _resize_args(sglang_model_num_engines=None)
    state = _semaphore_state_mock(value=128 * 6)
    sema_before = state.semaphore

    _resize_student_semaphore(args, state)

    assert state.semaphore is sema_before  # no resize when split is unknown
    assert state._tg_opd_semaphore_sized is True


@pytest.mark.unit
def test_resize_student_semaphore_noop_when_already_correct():
    args = _resize_args()
    state = _semaphore_state_mock(value=128 * 4)  # already the target
    sema_before = state.semaphore

    _resize_student_semaphore(args, state)

    assert state.semaphore is sema_before


# ---------------------------------------------------------------------------
# Timing instrumentation — _run_one_prompt timing dict + _emit_timing_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_one_prompt_returns_timing_dict_with_monotonic_stages(monkeypatch):
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]

    # Inject artificial sleeps so each stage has a measurable duration.
    async def slow_student(_args, group, sampling_params, evaluation):
        await asyncio.sleep(0.02)
        return _student_group(rewards)
    monkeypatch.setattr(tg, "generate_and_rm_group", slow_student)

    probes = [_sample(900, 1.0), _sample(901, 1.0), _sample(902, 0.0)]
    _install_teacher_probe(monkeypatch, probes)

    async def slow_logprob(_args, _ts, sample, _teacher_model_name):
        await asyncio.sleep(0.01)
        sample.teacher_log_probs = [0.5] * sample.response_length
    monkeypatch.setattr(tg, "_teacher_logprob", slow_logprob)

    _selected, _metrics, timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    # Schema check
    assert set(timing) >= {
        "t_start", "t_student_end", "t_logprob_start", "t_end",
        "student_wall", "logprob_wall", "prompt_total",
    }
    # Monotonic ordering
    assert timing["t_start"] <= timing["t_student_end"] <= timing["t_logprob_start"] <= timing["t_end"]
    # Stage durations match observed sleeps (with generous slack)
    assert timing["student_wall"] >= 0.015
    assert timing["logprob_wall"] >= 0.005
    assert timing["prompt_total"] == pytest.approx(timing["t_end"] - timing["t_start"])


@pytest.mark.unit
def test_run_one_prompt_online_fallback_follows_streaming_logprob(monkeypatch):
    args = _run_one_prompt_args()
    student_state = _student_state_mock()
    ts = MagicMock()
    rewards = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7]
    _install_student_generate(monkeypatch, lambda g: _student_group(rewards))
    # Online failure is applied after logprob; offline failures skip logprob.
    probes = [_sample(900, 1.0), _sample(901, 0.0), _sample(902, 0.0)]
    _install_teacher_probe(monkeypatch, probes)
    logprob_calls: list[int] = []
    _install_teacher_logprob(monkeypatch, logprob_calls)

    _selected, _metrics, timing = asyncio.run(_run_complete_prompt(args, student_state, ts, [_sample(0, 0.0)]))

    assert sorted(logprob_calls) == [0, 1, 4, 5]
    assert timing["logprob_wall"] >= 0.0
    assert all(sample.opd_mask == 0 for sample in _selected)


@pytest.mark.unit
def test_emit_timing_metrics_computes_distribution_and_timeline():
    """Three synthetic per-prompt timings; verify the per-stage distribution
    fields, the timeline offsets, and the derived overlap / after-student keys."""
    # origin = 100.0; prompt A: student [100..110], logprob [110..115], total 15
    #                 prompt B: student [101..120], logprob [120..125], total 24
    #                 prompt C: student [102..108], logprob [108..130], total 28
    timings = [
        {"t_start": 100.0, "t_student_end": 110.0, "t_logprob_start": 110.0, "t_end": 115.0,
         "student_wall": 10.0, "logprob_wall": 5.0, "prompt_total": 15.0},
        {"t_start": 101.0, "t_student_end": 120.0, "t_logprob_start": 120.0, "t_end": 125.0,
         "student_wall": 19.0, "logprob_wall": 5.0, "prompt_total": 24.0},
        {"t_start": 102.0, "t_student_end": 108.0, "t_logprob_start": 108.0, "t_end": 130.0,
         "student_wall": 6.0, "logprob_wall": 22.0, "prompt_total": 28.0},
    ]
    metrics: dict[str, float] = {}
    _emit_timing_metrics(timings, metrics)

    # Distribution: student_wall sorted = [6, 10, 19] -> median 10, max 19, p95 = idx round(0.95*2)=2 -> 19
    assert metrics["perf/tg_opd/timing/student_wall_s/mean"] == pytest.approx(35.0 / 3)
    assert metrics["perf/tg_opd/timing/student_wall_s/median"] == 10.0
    assert metrics["perf/tg_opd/timing/student_wall_s/max"] == 19.0
    assert metrics["perf/tg_opd/timing/student_wall_s/min"] == 6.0
    # logprob_wall sorted = [5, 5, 22]
    assert metrics["perf/tg_opd/timing/logprob_wall_s/max"] == 22.0
    assert metrics["perf/tg_opd/timing/logprob_wall_s/median"] == 5.0

    # Timeline (origin = min t_start = 100.0)
    # last_student_end  = max(110, 120, 108) - 100 = 20
    # first_logprob_start = min(110, 120, 108) - 100 = 8
    # last_logprob_end  = max(115, 125, 130) - 100 = 30
    assert metrics["perf/tg_opd/timeline/last_student_end_s"] == 20.0
    assert metrics["perf/tg_opd/timeline/first_logprob_start_s"] == 8.0
    assert metrics["perf/tg_opd/timeline/last_logprob_end_s"] == 30.0
    # logprob_after_student = 30 - 20 = 10 (wall-clock spent purely on logprob tail)
    assert metrics["perf/tg_opd/timeline/logprob_after_student_s"] == 10.0
    # overlap = 20 - 8 = 12 (logprob started early & ran while student was still going)
    assert metrics["perf/tg_opd/timeline/student_logprob_overlap_s"] == 12.0


@pytest.mark.unit
def test_emit_timing_metrics_empty_timings_noop():
    metrics: dict[str, float] = {"existing": 1.0}
    _emit_timing_metrics([], metrics)
    assert metrics == {"existing": 1.0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
