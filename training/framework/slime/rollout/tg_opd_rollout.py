"""TG-OPD rollout: keeps the student on slime's default rollout path and runs
teacher probe + teacher logprob as a side-channel.

Design rules (set by the user, do not violate):

* Student generation MUST go through ``slime.rollout.sglang_rollout.generate_and_rm_group``
  so sample-level SGLang calls, R3 routing-replay, session-id routing, reward
  calls, and semaphores stay on the stock student path. The driver mirrors the
  stock ``generate_rollout_async`` feeder loop: oversampling, dynamic filtering,
  replacement groups, and tail abort are preserved.
* Teacher is purely side-channel. A probe task races with the student and is
  consulted non-blocking after the student returns. The truth table is:

      probe done + pass_count >= min_pass  -> opd_mask=1 (OPD)
      probe done + pass_count <  min_pass  -> opd_mask=0 (GRPO fallback)
      probe NOT ready by student done      -> keep probe until batch end
      probe still not ready by batch end   -> opd_mask=1 (cancel probe, stay OPD)
      probe raised                         -> opd_mask=1 (conservative)
      tg_opd_mode == "selected_opd"        -> opd_mask=1 (no probe)

  OPD is the default; GRPO fallback only fires for prompts where the teacher
  has explicitly demonstrated unreliability.
* In ``offline_tg_opd`` mode the same reliability decision comes from a
  precomputed manifest keyed by prompt/label/metadata fingerprint. Missing
  records fail the rollout. Unreliable groups skip teacher logprob entirely.
* Teacher logprob is attached group-by-group as soon as a student group is
  accepted, instead of draining all teacher logprob requests after the whole
  student rollout batch completes. Teacher probe decisions are deferred until
  the batch end, so early-finishing student groups do not lose the chance to
  use a probe that finishes later in the same rollout batch.
* Teacher SGLang behaviour (engine count, mem_fraction, max_running_requests,
  ...) is fully controlled by ``--sglang-config``. The Python side only owns:
  the teacher semaphore cap (``--tg-opd-teacher-server-concurrency``), the
  probe max_new_tokens cap (``--tg-opd-probe-max-response-len``), and the
  probe top_p (``--tg-opd-probe-top-p``, default 1.0 so probe responses do
  not need top-p replay metadata).
* MOPD is an optional extension of the same path. When
  ``--mopd-teacher-routing`` is set, each prompt group routes to the frozen
  teacher named by ``sample.metadata[--mopd-domain-key]``. Without that flag,
  behaviour stays exactly single-teacher SOPD/TG-OPD.
"""

import asyncio
import copy
import logging
import os
import statistics
import time
import uuid
from argparse import Namespace
from collections import defaultdict
from contextlib import suppress
from typing import Any

import torch

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout import sglang_rollout
from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.rollout.offline_probe_manifest import OfflineProbeManifest
from slime.rollout.sglang_rollout import (
    GenerateState,
    eval_rollout,
    generate_and_rm_group,
    get_model_url,
)
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}


# ---------------------------------------------------------------------------
# Teacher-side state
# ---------------------------------------------------------------------------


class _TeacherState:
    """Lightweight singleton for teacher-only resources (tokenizer, semaphore,
    probe sampling params). Student state stays on
    :class:`slime.rollout.sglang_rollout.GenerateState`.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        self.teacher_model_name = args.tg_opd_teacher_model_name
        self.teacher_routing: dict[str, str] = getattr(args, "mopd_teacher_routing", None) or {}
        self.domain_key = getattr(args, "mopd_domain_key", "domain")
        self.probe_max_response_len_by_domain: dict[str, int] = (
            getattr(args, "tg_opd_probe_max_response_len_by_domain", None) or {}
        )
        self.offline_manifest = None
        if args.tg_opd_mode == "offline_tg_opd":
            self.offline_manifest = OfflineProbeManifest(
                args.tg_opd_offline_manifest,
                probe_n=args.tg_opd_probe_n,
                min_pass=args.tg_opd_probe_min_pass,
                reward_threshold=args.tg_opd_probe_pass_threshold,
                sample_k=getattr(args, "tg_opd_offline_sample_k", None),
                sample_min_pass=getattr(args, "tg_opd_offline_sample_min_pass", None),
                sample_seed=getattr(args, "tg_opd_offline_sample_seed", 42),
            )
            sample_k = getattr(args, "tg_opd_offline_sample_k", None)
            if sample_k is not None and sample_k < args.tg_opd_probe_n:
                logger.info(
                    "Loaded offline TG-OPD manifest %s with %s records; sub-sampled to K=%s min_pass=%s "
                    "(reliable %s / fallback %s after sub-sample).",
                    args.tg_opd_offline_manifest,
                    len(self.offline_manifest.records),
                    sample_k,
                    getattr(args, "tg_opd_offline_sample_min_pass", None),
                    self.offline_manifest.metadata.get("_reliable_count_after_subsample"),
                    self.offline_manifest.metadata.get("_fallback_count_after_subsample"),
                )
            else:
                logger.info(
                    "Loaded offline TG-OPD manifest %s with %s records.",
                    args.tg_opd_offline_manifest,
                    len(self.offline_manifest.records),
                )

        # Validation deferred to here (not arguments.validate_args) because
        # sglang_model_num_engines is populated by start_rollout_servers AFTER
        # parse-time, when the Ray actor starts.
        num_engines_by_model = getattr(args, "sglang_model_num_engines", None) or {}
        teacher_names = sorted(set(self.teacher_routing.values()) or {self.teacher_model_name})
        self.teacher_semaphores: dict[str, asyncio.Semaphore] = {}
        for teacher_name in teacher_names:
            num_engines = num_engines_by_model.get(teacher_name)
            if num_engines is None:
                raise ValueError(
                    f"TG-OPD teacher model {teacher_name!r} is not defined in "
                    f"--sglang-config. Available models: {sorted(num_engines_by_model.keys())}."
                )
            self.teacher_semaphores[teacher_name] = asyncio.Semaphore(
                args.tg_opd_teacher_server_concurrency * max(1, int(num_engines))
            )

        # Probe sampling mirrors student sampling EXCEPT max_new_tokens (capped
        # by --tg-opd-probe-max-response-len) and top_p (forced to
        # --tg-opd-probe-top-p, default 1.0). Skipping custom_params avoids the
        # return_top_p_token_ids branch in Sample._apply_meta_info.
        self.probe_sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.tg_opd_probe_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.tg_opd_probe_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

    def probe_sampling_params_for(self, sample: Sample) -> dict[str, Any]:
        sampling_params = self.probe_sampling_params.copy()
        domain = (sample.metadata or {}).get(self.domain_key)
        if domain in self.probe_max_response_len_by_domain:
            sampling_params["max_new_tokens"] = self.probe_max_response_len_by_domain[domain]
        return sampling_params

    def domain_for(self, sample: Sample) -> str | None:
        if not self.teacher_routing:
            return None
        metadata = sample.metadata or {}
        domain = metadata.get(self.domain_key)
        if domain is None:
            raise ValueError(
                f"MOPD routing requires sample.metadata[{self.domain_key!r}], "
                f"but sample index={sample.index} has metadata keys={sorted(metadata.keys())}."
            )
        return str(domain)

    def teacher_for(self, sample: Sample) -> str:
        domain = self.domain_for(sample)
        if domain is None:
            return self.teacher_model_name
        try:
            return self.teacher_routing[domain]
        except KeyError as exc:
            raise ValueError(
                f"MOPD routing has no teacher for domain {domain!r}. "
                f"Known domains: {sorted(self.teacher_routing.keys())}."
            ) from exc

    def semaphore_for(self, teacher_model_name: str) -> asyncio.Semaphore:
        return self.teacher_semaphores[teacher_model_name]


_teacher_state_cache: _TeacherState | None = None


def _get_teacher_state(args: Namespace) -> _TeacherState:
    global _teacher_state_cache
    if _teacher_state_cache is None or _teacher_state_cache.args is not args:
        _teacher_state_cache = _TeacherState(args)
    return _teacher_state_cache


def _resize_student_semaphore(args: Namespace, student_state: GenerateState) -> None:
    """Resize the student SGLang semaphore to match the student engine count.

    GenerateState sizes its semaphore from ``get_rollout_num_engines(args)``
    which divides ``args.rollout_num_gpus`` (total student + teacher GPUs in
    the --sglang-config split layout) by ``rollout_num_gpus_per_engine``. With
    e.g. 16 student GPU + 8 teacher GPU + 4 GPU/engine, that yields 6 engines
    of permitted concurrency for a student router that only fronts 4 engines
    — the extra 50% queues at the SGLang router beyond its
    ``max_running_requests`` cap and hurts long-tail latency, which then bleeds
    into student rollout wall-clock.

    Fix: once per driver lifetime, replace student_state.semaphore with one
    sized by args.sglang_model_num_engines[args.tg_opd_student_model_name].
    Idempotent via a sentinel attribute so repeated rollouts do not churn.

    The connection-pool size used by ``init_http_client`` is left alone — it
    spans student AND teacher HTTP and needs the total-engine budget.
    """
    if getattr(student_state, "_tg_opd_semaphore_sized", False):
        return
    num_engines_by_model = getattr(args, "sglang_model_num_engines", None) or {}
    student_engines = num_engines_by_model.get(args.tg_opd_student_model_name)
    if student_engines is None:
        # No per-model split known (single-model deployment); the default
        # GenerateState sizing is already correct in that case.
        student_state._tg_opd_semaphore_sized = True
        return
    desired = args.sglang_server_concurrency * max(1, int(student_engines))
    old_value = getattr(student_state.semaphore, "_value", None)
    if old_value != desired:
        logger.info(
            "TG-OPD resizing student semaphore: %s -> %s "
            "(student_engines=%s, sglang_server_concurrency=%s).",
            old_value,
            desired,
            student_engines,
            args.sglang_server_concurrency,
        )
        student_state.semaphore = asyncio.Semaphore(desired)
    student_state._tg_opd_semaphore_sized = True


# ---------------------------------------------------------------------------
# Small helpers (kept verbatim where possible)
# ---------------------------------------------------------------------------


def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    """Mirror of ``slime.rollout.sglang_rollout._prepare_prompt_ids`` (which is
    private). Used only for teacher probe payloads so the probe sees the same
    prompt token sequence as the student.
    """
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (
        sample.multimodal_train_inputs is not None or not has_multimodal_inputs
    )

    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {
                k: v for k, v in processor_output.items() if k not in _PROCESSOR_PROMPT_KEYS
            } or None
        return prompt_ids

    if reuse_existing_input_ids:
        return sample.tokens

    return tokenizer.encode(sample.prompt, add_special_tokens=False)


def _reset_to_prompt(sample: Sample) -> Sample:
    """Deep-copy a prompt sample and clear response-side fields so the copy can
    be used as an independent teacher probe. Regenerates session_id so probes
    do not collide with student consistent-hashing routing keys.
    """
    sample = copy.deepcopy(sample)
    sample.tokens = []
    sample.response = ""
    sample.response_length = 0
    sample.reward = None
    sample.loss_mask = None
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    sample.teacher_log_probs = None
    sample.opd_mask = None
    sample.status = Sample.Status.PENDING
    sample.session_id = str(uuid.uuid4())
    return sample


async def _cancel_task(task: asyncio.Task) -> None:
    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _select_student_samples(args: Namespace, group: list[Sample]) -> list[Sample]:
    """Pick ``tg_opd_select_k`` samples from a student rollout group. Strategy
    ``high_low`` keeps ``k//2`` lowest-reward and ``k-k//2`` highest-reward
    samples, in their original group order; ``top``/``low`` are obvious.
    """
    k = args.tg_opd_select_k
    ranked = sorted(enumerate(group), key=lambda item: item[1].get_reward_value(args))
    if args.tg_opd_select_strategy == "top":
        selected_positions = {pos for pos, _ in ranked[-k:]}
    elif args.tg_opd_select_strategy == "low":
        selected_positions = {pos for pos, _ in ranked[:k]}
    elif args.tg_opd_select_strategy == "high_low":
        low_count = k // 2
        high_count = k - low_count
        selected_positions = {pos for pos, _ in ranked[:low_count]}
        selected_positions.update(pos for pos, _ in ranked[-high_count:])
    else:
        raise ValueError(f"Unsupported TG-OPD select strategy: {args.tg_opd_select_strategy}")
    return [sample for pos, sample in enumerate(group) if pos in selected_positions]


def _probe_pass_count(args: Namespace, probes: list[Sample]) -> int:
    threshold = args.tg_opd_probe_pass_threshold
    return sum(1 for sample in probes if sample.get_reward_value(args) >= threshold)


def _unreliable_teacher_action(args: Namespace) -> str:
    action = os.environ.get("MOPD_UNRELIABLE_ACTION")
    if action:
        action = action.strip().lower()
        if action in {"zero_teacher_signal", "mask", "grpo"}:
            return "zero_teacher_signal" if action == "mask" else action
        raise ValueError(f"Unsupported MOPD_UNRELIABLE_ACTION={action!r}")

    action = getattr(args, "tg_opd_unreliable_action", "grpo")
    return "zero_teacher_signal" if action == "mask" else action


def _probe_not_ready_assumes_reliable() -> bool:
    action = os.environ.get("MOPD_PROBE_NOT_READY_ACTION")
    if not action:
        return True
    action = action.strip().lower()
    if action in {"assume_reliable", "legacy"}:
        return True
    if action in {"mark_unreliable", "grpo"}:
        return False
    raise ValueError(f"Unsupported MOPD_PROBE_NOT_READY_ACTION={action!r}")


def _safe_metric_key(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _sample_data_source(sample: Sample) -> str | None:
    metadata = sample.metadata or {}
    data_source = metadata.get("data_source")
    if data_source is None:
        reward_spec = metadata.get("reward_spec") or {}
        if isinstance(reward_spec, dict):
            data_source = reward_spec.get("data_source")
    return str(data_source) if data_source is not None else None


def _add_mopd_metric(
    metrics: dict[str, float],
    suffix: str,
    value: float,
    domain: str | None = None,
    data_source: str | None = None,
) -> None:
    metrics[f"mopd/{suffix}"] = metrics.get(f"mopd/{suffix}", 0.0) + value
    if domain is not None:
        metrics[f"mopd/domain/{_safe_metric_key(domain)}/{suffix}"] = (
            metrics.get(f"mopd/domain/{_safe_metric_key(domain)}/{suffix}", 0.0) + value
        )
    if data_source is not None:
        metrics[f"mopd/data_source/{_safe_metric_key(data_source)}/{suffix}"] = (
            metrics.get(f"mopd/data_source/{_safe_metric_key(data_source)}/{suffix}", 0.0) + value
        )


def _finalize_mopd_rates(metrics: dict[str, Any]) -> None:
    prefixes = {"mopd"}
    for key in list(metrics.keys()):
        for marker in ("/teacher_verify/total_count", "/teacher_signal/total_count"):
            if key.endswith(marker):
                prefixes.add(key[: -len(marker)])

    for prefix in prefixes:
        verify_total = metrics.get(f"{prefix}/teacher_verify/total_count", 0.0)
        if verify_total > 0:
            metrics[f"{prefix}/teacher_verify/reliable_rate"] = (
                metrics.get(f"{prefix}/teacher_verify/reliable_count", 0.0) / verify_total
            )
            metrics[f"{prefix}/teacher_verify/unreliable_rate"] = (
                metrics.get(f"{prefix}/teacher_verify/unreliable_count", 0.0) / verify_total
            )
            metrics[f"{prefix}/teacher_verify/not_ready_rate"] = (
                metrics.get(f"{prefix}/teacher_verify/not_ready_count", 0.0) / verify_total
            )

        signal_total = metrics.get(f"{prefix}/teacher_signal/total_count", 0.0)
        if signal_total > 0:
            zeroed = metrics.get(f"{prefix}/teacher_signal/zeroed_count", 0.0)
            metrics[f"{prefix}/teacher_signal/zeroed_rate"] = zeroed / signal_total
            metrics[f"{prefix}/teacher_signal/effective_rate"] = 1.0 - zeroed / signal_total


def _accumulate_teacher_signal_metrics(metrics: dict[str, Any], groups: list[list[Sample]]) -> None:
    for group in groups:
        for sample in group:
            metadata = sample.metadata or {}
            domain = metadata.get("domain")
            data_source = _sample_data_source(sample)
            _add_mopd_metric(metrics, "teacher_signal/total_count", 1.0, domain, data_source)
            if _sample_is_loss_masked(sample):
                _add_mopd_metric(metrics, "teacher_signal/zeroed_count", 1.0, domain, data_source)


def _zero_sample_training_signal(sample: Sample) -> None:
    sample.loss_mask = [0] * sample.response_length


def _disable_opd_signal(sample: Sample) -> None:
    sample.teacher_log_probs = [0.0] * sample.response_length
    sample.opd_mask = 0


def _apply_unreliable_teacher_action(args: Namespace, sample: Sample) -> None:
    _disable_opd_signal(sample)
    if _unreliable_teacher_action(args) == "zero_teacher_signal":
        _zero_sample_training_signal(sample)


def _sample_is_loss_masked(sample: Sample) -> bool:
    return sample.loss_mask is not None and sum(sample.loss_mask) == 0
# ---------------------------------------------------------------------------
# Teacher-side I/O (probe rollouts, logprob prefill)
# ---------------------------------------------------------------------------


async def _teacher_generate(
    args: Namespace, ts: _TeacherState, sample: Sample, teacher_model_name: str
) -> Sample:
    """Single teacher rollout against the teacher SGLang router.

    Bypasses :func:`slime.rollout.sglang_rollout.generate` because that
    function hardcodes the default-router URL; here we route to the teacher
    via ``get_model_url(args, teacher_model_name)``.
    """
    assert sample.status in (Sample.Status.PENDING, Sample.Status.ABORTED), (
        f"Probe sample status is {sample.status}"
    )
    sampling_params = ts.probe_sampling_params_for(sample)
    assert sampling_params["max_new_tokens"] >= 0
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    prompt_ids = _prepare_prompt_ids(sample, ts.tokenizer, ts.processor)
    payload: dict[str, Any] = {
        "rid": uuid.uuid4().hex,
        "sampling_params": sampling_params,
        # return_logprob is required by Sample.append_response_tokens for trainable
        # response tokens (types.py rejects empty log_probs when tokens are non-empty);
        # the cost is small on the teacher side.
        "return_logprob": True,
    }

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    async with ts.semaphore_for(teacher_model_name):
        url = get_model_url(args, teacher_model_name)
        with trace_span(sample, "tg_opd_teacher_probe", attrs={"model": teacher_model_name}) as span:
            output = await post(url, payload)
            span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        # Should not happen because return_logprob=True, but guard anyway.
        new_response_tokens, new_response_log_probs = [], []

    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )
    return sample


async def _teacher_probe(
    args: Namespace, ts: _TeacherState, prompt_sample: Sample, teacher_model_name: str
) -> list[Sample]:
    """Run ``args.tg_opd_probe_n`` teacher rollouts on the prompt and score them.

    Returns the scored probe samples; the caller computes ``pass_count`` via
    :func:`_probe_pass_count`.
    """
    probes = [_reset_to_prompt(prompt_sample) for _ in range(args.tg_opd_probe_n)]
    tasks = [asyncio.create_task(_teacher_generate(args, ts, sample, teacher_model_name)) for sample in probes]
    try:
        generated = await asyncio.gather(*tasks)
    except BaseException:
        # If gather raises (cancellation or single-task failure), cancel any
        # siblings that are still running so we do not leak teacher HTTP work.
        for t in tasks:
            if not t.done():
                t.cancel()
        raise

    if args.group_rm:
        rewards = await batched_async_rm(args, generated)
    else:
        rewards = await asyncio.gather(*[async_rm(args, sample) for sample in generated])
    for sample, reward in zip(generated, rewards, strict=True):
        sample.reward = reward
    return generated


async def _teacher_logprob(
    args: Namespace, ts: _TeacherState, sample: Sample, teacher_model_name: str
) -> None:
    """Prefill-only teacher call computing log p_teacher(student_response).

    Writes ``sample.teacher_log_probs`` in place. Raises on transport failure;
    the caller (:func:`_run_one_prompt`) is responsible for per-sample
    fallback so one failure does not contaminate the rest of the group.
    """
    payload: dict[str, Any] = {
        "rid": uuid.uuid4().hex,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]

    async with ts.semaphore_for(teacher_model_name):
        url = get_model_url(args, teacher_model_name)
        with trace_span(sample, "tg_opd_teacher_logprob", attrs={"model": teacher_model_name}) as span:
            output = await post(url, payload)
            span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    # First input token has no preceding context, so drop it. Then take the
    # trailing `response_length` entries which correspond to teacher logp
    # over the student response tokens.
    input_logprobs = output["meta_info"]["input_token_logprobs"][1:]
    teacher_log_probs = (
        []
        if sample.response_length == 0
        else [item[0] for item in input_logprobs][-sample.response_length :]
    )
    assert len(teacher_log_probs) == sample.response_length, (
        f"teacher_log_probs length {len(teacher_log_probs)} != response_length {sample.response_length}"
    )
    sample.teacher_log_probs = teacher_log_probs


# ---------------------------------------------------------------------------
# Per-prompt orchestration
# ---------------------------------------------------------------------------


async def _run_one_prompt(
    args: Namespace,
    student_state: GenerateState,
    ts: _TeacherState,
    group: list[Sample],
    dynamic_filter: Any | None,
) -> tuple[list[Sample], dict[str, float], dict[str, float], bool, str | None, dict[str, Any] | None]:
    """Run one prompt group with streaming logprob and deferred probe gate.

    Student generation/reward still uses the stock ``generate_and_rm_group``.
    Accepted groups immediately get teacher logprob with ``opd_mask=1`` as the
    default OPD path. The teacher probe task is returned to the driver and is
    judged at rollout-batch end; only a completed-and-failed probe flips the
    selected group to GRPO fallback.
    """
    metrics: dict[str, float] = {}
    t_start = time.monotonic()
    prompt_sample = group[0]
    domain = ts.domain_for(prompt_sample)
    data_source = _sample_data_source(prompt_sample)
    teacher_model_name = ts.teacher_for(prompt_sample)
    if domain is not None:
        metrics[f"mopd/{domain}/groups"] = 1.0
    _add_mopd_metric(metrics, "groups", 1.0, domain, data_source)

    offline_reliable: bool | None = None
    if args.tg_opd_mode == "offline_tg_opd":
        assert ts.offline_manifest is not None
        offline_record = ts.offline_manifest.lookup(prompt_sample)
        offline_reliable = bool(offline_record["reliable"])
        pass_count = int(offline_record["pass_count"])
        metrics["tg_opd/offline_lookup_hits"] = 1.0
        metrics["tg_opd/offline_pass_count"] = float(pass_count)
        metrics[f"tg_opd/offline_pass_count_{pass_count}"] = 1.0
        _add_mopd_metric(metrics, "teacher_verify/total_count", 1.0, domain, data_source)
        _add_mopd_metric(
            metrics,
            "teacher_verify/reliable_count" if offline_reliable else "teacher_verify/unreliable_count",
            1.0,
            domain,
            data_source,
        )

    probe_record: dict[str, Any] | None = None
    probe_task: asyncio.Task | None = None
    if args.tg_opd_mode == "tg_opd":
        probe_record = {
            "t_start": t_start,
            "finish_time": None,
            "domain": domain,
            "data_source": data_source,
            "teacher_model_name": teacher_model_name,
        }

        def _record_probe_finish(_task: asyncio.Task) -> None:
            if probe_record is not None:
                probe_record["finish_time"] = time.monotonic()

        probe_task = asyncio.create_task(_teacher_probe(args, ts, prompt_sample, teacher_model_name))
        probe_task.add_done_callback(_record_probe_finish)
        probe_record["task"] = probe_task

    try:
        student_group = await generate_and_rm_group(
            args,
            group,
            sampling_params=student_state.sampling_params.copy(),
            evaluation=False,
        )
    except BaseException:
        if probe_task is not None:
            await _cancel_task(probe_task)
        raise

    t_student_end = time.monotonic()
    assert len(student_group) == args.n_samples_per_prompt

    dynamic_filter_output = sglang_rollout.call_dynamic_filter(dynamic_filter, args, student_group)
    if not dynamic_filter_output.keep:
        if probe_task is not None:
            await _cancel_task(probe_task)
        timing = {
            "t_start": t_start,
            "t_student_end": t_student_end,
            "t_logprob_start": t_student_end,
            "t_end": t_student_end,
            "student_wall": t_student_end - t_start,
            "logprob_wall": 0.0,
            "prompt_total": t_student_end - t_start,
        }
        return [], metrics, timing, False, dynamic_filter_output.reason, None

    selected = _select_student_samples(args, student_group)

    if offline_reliable is False:
        for sample in selected:
            _apply_unreliable_teacher_action(args, sample)
        metrics["tg_opd/fallback_groups"] = 1.0
        metrics["tg_opd/teacher_logprob_skipped_groups"] = 1.0
        if domain is not None:
            metrics[f"mopd/{domain}/fallback_groups"] = 1.0
        t_end = time.monotonic()
        timing = {
            "t_start": t_start,
            "t_student_end": t_student_end,
            "t_logprob_start": t_end,
            "t_end": t_end,
            "student_wall": t_student_end - t_start,
            "logprob_wall": 0.0,
            "prompt_total": t_end - t_start,
        }
        return selected, metrics, timing, True, None, None

    # Default path is OPD: compute teacher logprob immediately. The deferred
    # probe gate may later apply the configured unreliable-teacher action.
    t_logprob_start = time.monotonic()
    results = await asyncio.gather(
        *[_teacher_logprob(args, ts, s, teacher_model_name) for s in selected],
        return_exceptions=True,
    )
    for s, r in zip(selected, results, strict=True):
        if isinstance(r, BaseException):
            logger.warning(
                "tg_opd teacher_logprob failed for sample %s; applying unreliable teacher action.",
                s.index,
                exc_info=r,
            )
            _apply_unreliable_teacher_action(args, s)
            metrics["tg_opd/teacher_logprob_failures"] = (
                metrics.get("tg_opd/teacher_logprob_failures", 0.0) + 1.0
            )
            if domain is not None:
                metrics[f"mopd/{domain}/teacher_logprob_failures"] = (
                    metrics.get(f"mopd/{domain}/teacher_logprob_failures", 0.0) + 1.0
                )
        else:
            s.opd_mask = 1

    if args.tg_opd_mode in {"selected_opd", "offline_tg_opd"}:
        metrics["tg_opd/opd_groups"] = 1.0
        if domain is not None:
            metrics[f"mopd/{domain}/opd_groups"] = 1.0

    t_end = time.monotonic()
    timing: dict[str, float] = {
        "t_start": t_start,
        "t_student_end": t_student_end,
        "t_logprob_start": t_logprob_start,
        "t_end": t_end,
        "student_wall": t_student_end - t_start,
        "logprob_wall": t_end - t_logprob_start,
        "prompt_total": t_end - t_start,
    }
    if domain is not None:
        metrics[f"mopd/{domain}/logprob_wall_s"] = t_end - t_logprob_start
    if probe_record is not None:
        probe_record["t_student_end"] = t_student_end
        probe_record["selected"] = selected
        probe_record["timing"] = timing
    return selected, metrics, timing, True, None, probe_record


async def _finalize_deferred_teacher_gate(
    args: Namespace,
    probe_records: list[dict[str, Any]],
    metrics_acc: dict[str, float],
) -> None:
    """Apply the teacher reliability gate at rollout-batch end.

    This is deliberately non-blocking: completed probes are consumed; probes
    still in flight are canceled and keep the default OPD signal already
    produced by streaming teacher logprob.
    """
    if args.tg_opd_mode != "tg_opd":
        return

    batch_gate_finished = 0.0
    for record in probe_records:
        task: asyncio.Task = record["task"]
        selected: list[Sample] = record["selected"]
        timing: dict[str, float] = record["timing"]
        domain = record.get("domain")
        data_source = record.get("data_source")
        reliable = True
        _add_mopd_metric(metrics_acc, "teacher_verify/total_count", 1.0, domain, data_source)

        if task.done():
            batch_gate_finished += 1.0
            finish_time = record.get("finish_time") or time.monotonic()
            timing["probe_latency"] = max(0.0, finish_time - record["t_start"])
            timing["probe_finish_before_student"] = 1.0 if finish_time <= record["t_student_end"] else 0.0
            timing["probe_finish_before_batch_end"] = 1.0
            try:
                probes = task.result()
                pass_count = _probe_pass_count(args, probes)
                reliable = pass_count >= args.tg_opd_probe_min_pass
                metrics_acc["tg_opd/probe_completed"] += 1.0
                metrics_acc["tg_opd/probe_pass_count"] += float(pass_count)
                _add_mopd_metric(
                    metrics_acc,
                    "teacher_verify/reliable_count" if reliable else "teacher_verify/unreliable_count",
                    1.0,
                    domain,
                    data_source,
                )
                if domain is not None:
                    metrics_acc[f"mopd/{domain}/probe_completed"] += 1.0
                    metrics_acc[f"mopd/{domain}/probe_pass_count"] += float(pass_count)
            except asyncio.CancelledError:
                metrics_acc["tg_opd/probe_timeout"] += 1.0
                _add_mopd_metric(metrics_acc, "teacher_verify/not_ready_count", 1.0, domain, data_source)
                if domain is not None:
                    metrics_acc[f"mopd/{domain}/probe_timeout"] += 1.0
                reliable = _probe_not_ready_assumes_reliable()
            except Exception:
                logger.exception("Teacher probe raised; defaulting to OPD (reliable=True).")
                metrics_acc["tg_opd/probe_errored"] += 1.0
                _add_mopd_metric(metrics_acc, "teacher_verify/not_ready_count", 1.0, domain, data_source)
                _add_mopd_metric(metrics_acc, "teacher_verify/errored_count", 1.0, domain, data_source)
                if domain is not None:
                    metrics_acc[f"mopd/{domain}/probe_errored"] += 1.0
                reliable = _probe_not_ready_assumes_reliable()
        else:
            timing["probe_finish_before_student"] = 0.0
            timing["probe_finish_before_batch_end"] = 0.0
            await _cancel_task(task)
            metrics_acc["tg_opd/probe_timeout"] += 1.0
            _add_mopd_metric(metrics_acc, "teacher_verify/not_ready_count", 1.0, domain, data_source)
            if domain is not None:
                metrics_acc[f"mopd/{domain}/probe_timeout"] += 1.0
            reliable = _probe_not_ready_assumes_reliable()

        if reliable:
            metrics_acc["tg_opd/opd_groups"] += 1.0
            if domain is not None:
                metrics_acc[f"mopd/{domain}/opd_groups"] += 1.0
        else:
            metrics_acc["tg_opd/fallback_groups"] += 1.0
            if domain is not None:
                metrics_acc[f"mopd/{domain}/fallback_groups"] += 1.0
            for sample in selected:
                _apply_unreliable_teacher_action(args, sample)

    metrics_acc["tg_opd/probe_finished_by_batch_end"] += batch_gate_finished

# ---------------------------------------------------------------------------
# Driver loop
# ---------------------------------------------------------------------------


class _TeacherProbeDataSource:
    """Thin side-channel around the stock data source.

    The wrapped ``get_samples`` call is passed to the stock student rollout
    driver. That driver still owns all student scheduling, filtering, abort, and
    replacement semantics. This wrapper only starts teacher probes for the same
    prompt groups as soon as the stock feeder fetches them.
    """

    def __init__(self, args: Namespace, ts: _TeacherState, data_source: Any, rollout_start: float) -> None:
        self.args = args
        self.ts = ts
        self.data_source = data_source
        self.rollout_start = rollout_start
        self.probe_tasks: dict[int, asyncio.Task] = {}
        self.probe_start_times: dict[int, float] = {}

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = self.data_source.get_samples(num_samples)
        if self.args.tg_opd_mode != "tg_opd":
            return groups

        for group in groups:
            if not group:
                continue
            group_index = group[0].group_index
            if group_index is None or group_index in self.probe_tasks:
                continue
            teacher_model_name = self.ts.teacher_for(group[0])
            self.probe_start_times[group_index] = time.monotonic()
            self.probe_tasks[group_index] = asyncio.create_task(
                _teacher_probe(self.args, self.ts, group[0], teacher_model_name)
            )
        return groups

    async def pop_probe(self, group_index: int | None) -> tuple[str, list[Sample] | None]:
        if group_index is None:
            return "missing", None
        task = self.probe_tasks.pop(group_index, None)
        if task is None:
            return "missing", None
        if task.done():
            try:
                return "completed", task.result()
            except asyncio.CancelledError:
                return "timeout", None
            except Exception:
                logger.exception("Teacher probe raised; defaulting to OPD (reliable=True).")
                return "errored", None

        await _cancel_task(task)
        return "timeout", None

    def start_time(self, group_index: int | None) -> float:
        if group_index is None:
            return self.rollout_start
        return self.probe_start_times.get(group_index, self.rollout_start)

    async def cancel_pending(self) -> None:
        pending = [task for task in self.probe_tasks.values() if not task.done()]
        await asyncio.gather(*(_cancel_task(task) for task in pending), return_exceptions=True)
        self.probe_tasks.clear()


async def _select_and_attach_teacher_signal(
    args: Namespace,
    ts: _TeacherState,
    student_groups: list[list[Sample]],
    probe_data_source: _TeacherProbeDataSource,
    base_metrics: dict[str, Any] | None,
) -> RolloutFnTrainOutput:
    selected_groups: list[list[Sample]] = []
    accepted_timings: list[dict[str, float]] = []
    metrics_acc: dict[str, float] = defaultdict(float)
    metrics: dict[str, Any] = dict(base_metrics or {})

    # The stock driver returns only after the accepted student rollout batch is
    # complete. We keep this as the student fence; teacher logprob timing below
    # therefore measures the post-student drain conservatively.
    t_student_end = time.monotonic()
    do_print = True

    for student_group in student_groups:
        group_index = student_group[0].group_index if student_group else None
        t_start = probe_data_source.start_time(group_index)
        teacher_model_name = ts.teacher_for(student_group[0])
        selected = _select_student_samples(args, student_group)
        selected_groups.append(selected)

        reliable = True
        if args.tg_opd_mode == "tg_opd":
            probe_status, probes = await probe_data_source.pop_probe(group_index)
            if probe_status == "completed" and probes is not None:
                pass_count = _probe_pass_count(args, probes)
                reliable = pass_count >= args.tg_opd_probe_min_pass
                metrics_acc["tg_opd/probe_completed"] += 1.0
                metrics_acc["tg_opd/probe_pass_count"] += float(pass_count)
            elif probe_status == "timeout":
                metrics_acc["tg_opd/probe_timeout"] += 1.0
                # reliable stays True: probe not ready means default OPD.
            elif probe_status == "errored":
                metrics_acc["tg_opd/probe_errored"] += 1.0
                # reliable stays True: probe failure means default OPD.

        metrics_acc["tg_opd/opd_groups"] += 1.0 if reliable else 0.0
        metrics_acc["tg_opd/fallback_groups"] += 0.0 if reliable else 1.0

        t_logprob_start = time.monotonic()
        if reliable:
            results = await asyncio.gather(
                *[_teacher_logprob(args, ts, s, teacher_model_name) for s in selected],
                return_exceptions=True,
            )
            for sample, result in zip(selected, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning(
                        "tg_opd teacher_logprob failed for sample %s; applying unreliable teacher action.",
                        sample.index,
                        exc_info=result,
                    )
                    _apply_unreliable_teacher_action(args, sample)
                    metrics_acc["tg_opd/teacher_logprob_failures"] += 1.0
                else:
                    sample.opd_mask = 1
        else:
            for sample in selected:
                _apply_unreliable_teacher_action(args, sample)

        t_end = time.monotonic()
        accepted_timings.append(
            {
                "t_start": t_start,
                "t_student_end": t_student_end,
                "t_logprob_start": t_logprob_start,
                "t_end": t_end,
                "student_wall": t_student_end - t_start,
                "logprob_wall": (t_end - t_logprob_start) if reliable else 0.0,
                "prompt_total": t_end - t_start,
            }
        )

        if do_print:
            sample = selected[0]
            logger.info(
                "First TG-OPD rollout sample: %s, label: %s, reward: %s, opd_mask: %s",
                [str(sample.prompt) + sample.response],
                str(sample.label)[:100],
                sample.reward,
                sample.opd_mask,
            )
            do_print = False

    num_groups = max(1, len(selected_groups))
    for key, value in metrics_acc.items():
        metrics[key] = metrics.get(key, 0.0) + value
    metrics["tg_opd/opd_rate"] = metrics.get("tg_opd/opd_groups", 0.0) / num_groups
    metrics["tg_opd/fallback_rate"] = metrics.get("tg_opd/fallback_groups", 0.0) / num_groups
    metrics["tg_opd/masked_samples"] = float(sum(1 for group in selected_groups for sample in group if _sample_is_loss_masked(sample)))
    metrics["tg_opd/masked_sample_rate"] = metrics["tg_opd/masked_samples"] / max(1, sum(len(group) for group in selected_groups))
    metrics["tg_opd/masked_groups"] = float(sum(1 for group in selected_groups if group and all(_sample_is_loss_masked(sample) for sample in group)))
    metrics["tg_opd/masked_group_rate"] = metrics["tg_opd/masked_groups"] / num_groups
    metrics["tg_opd/probe_completed_rate"] = metrics.get("tg_opd/probe_completed", 0.0) / num_groups
    metrics["tg_opd/probe_timeout_rate"] = metrics.get("tg_opd/probe_timeout", 0.0) / num_groups
    metrics["tg_opd/probe_errored_rate"] = metrics.get("tg_opd/probe_errored", 0.0) / num_groups
    if args.tg_opd_mode == "tg_opd" and metrics["tg_opd/opd_rate"] == 0:
        logger.warning(
            "All TG-OPD prompts used the unreliable-teacher action. Check "
            "--tg-opd-probe-pass-threshold and the teacher reward signal."
        )

    _emit_timing_metrics(accepted_timings, metrics)
    return RolloutFnTrainOutput(samples=selected_groups, metrics=metrics)


async def _generate_rollout_async(args: Namespace, rollout_id: int, data_source) -> RolloutFnTrainOutput:
    """Run TG-OPD with group-level streaming teacher logprob.

    This mirrors the stock ``sglang_rollout.generate_rollout_async`` feeder
    loop, but each completed/accepted student group is selected and sent to the
    teacher immediately. That removes the previous batch-end teacher-logprob
    drain while keeping replacement-group and tail-abort behaviour.
    """
    assert args.rollout_global_dataset
    logger.info("TG-OPD rollout %s starting (mode=%s, streaming_logprob=True)", rollout_id, args.tg_opd_mode)

    student_state = GenerateState(args)
    _resize_student_semaphore(args, student_state)
    ts = _get_teacher_state(args)

    dynamic_filter = (
        sglang_rollout.load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )
    metric_gatherer = sglang_rollout.MetricGatherer()

    target_data_size = args.rollout_batch_size
    data: list[list[Sample]] = []
    all_data: list[list[Sample]] = []
    accepted_timings: list[dict[str, float]] = []
    probe_records: list[dict[str, Any]] = []
    metrics_acc: dict[str, float] = defaultdict(float)
    do_print = True

    pbar = sglang_rollout.tqdm(total=target_data_size * args.tg_opd_select_k, desc="TG-OPD rollout")
    try:
        while len(data) < target_data_size:
            while student_state.remaining_batch_size < target_data_size:
                samples = data_source.get_samples(args.over_sampling_batch_size)
                for group in samples:
                    student_state.pendings.add(
                        asyncio.create_task(
                            _run_one_prompt(args, student_state, ts, group, dynamic_filter)
                        )
                    )
                student_state.remaining_batch_size += len(samples)

            done, student_state.pendings = await asyncio.wait(
                student_state.pendings,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                selected, prompt_metrics, timing, keep, drop_reason, probe_record = task.result()

                if not keep:
                    metric_gatherer.on_dynamic_filter_drop(reason=drop_reason)
                    student_state.remaining_batch_size -= 1
                    continue

                assert len(selected) == args.tg_opd_select_k
                all_data.append(selected)

                if len(data) < target_data_size:
                    data.append(selected)
                    accepted_timings.append(timing)
                    if probe_record is not None:
                        probe_records.append(probe_record)
                    for key, value in prompt_metrics.items():
                        metrics_acc[key] += value
                    pbar.update(len(selected))

                    if do_print:
                        sample = selected[0]
                        logger.info(
                            "First TG-OPD rollout sample: %s, label: %s, reward: %s, opd_mask: %s",
                            [str(sample.prompt) + sample.response],
                            str(sample.label)[:100],
                            sample.reward,
                            sample.opd_mask,
                        )
                        do_print = False

        pbar.close()
        sample = data[-1][0]
        logger.info(
            "Finish TG-OPD rollout: %s, label: %s, reward: %s, opd_mask: %s",
            [str(sample.prompt) + sample.response],
            str(sample.label)[:100],
            sample.reward,
            sample.opd_mask,
        )

        # Preserve stock tail-abort semantics. The pending tasks are TG-OPD
        # tuples, but stock abort only inspects task.result() when
        # partial_rollout=True; this launcher uses partial_rollout=False.
        aborted_samples = await sglang_rollout.abort(args, rollout_id)
        if aborted_samples:
            data_source.add_samples(aborted_samples)

        await _finalize_deferred_teacher_gate(args, probe_records, metrics_acc)

        assert len(data) == args.rollout_batch_size, (
            f"Got {len(data)} groups, expected {args.rollout_batch_size}"
        )
        data = sorted(data, key=lambda group: group[0].index)
        all_samples = sorted(all_data, key=lambda group: group[0].index)

        student_state.reset()
        if args.rollout_sample_filter_path is not None:
            filter_func = sglang_rollout.load_function(args.rollout_sample_filter_path)
            filter_func(args, data)

        if args.rollout_all_samples_process_path is not None:
            process_func = sglang_rollout.load_function(args.rollout_all_samples_process_path)
            process_func(args, all_samples, data_source)

        metrics: dict[str, Any] = metric_gatherer.collect()
        num_groups = max(1, len(data))
        for key, value in metrics_acc.items():
            metrics[key] = metrics.get(key, 0.0) + value
        metrics["tg_opd/opd_rate"] = metrics.get("tg_opd/opd_groups", 0.0) / num_groups
        metrics["tg_opd/fallback_rate"] = metrics.get("tg_opd/fallback_groups", 0.0) / num_groups
        metrics["tg_opd/masked_samples"] = float(sum(1 for group in data for sample in group if _sample_is_loss_masked(sample)))
        metrics["tg_opd/masked_sample_rate"] = metrics["tg_opd/masked_samples"] / max(1, sum(len(group) for group in data))
        metrics["tg_opd/masked_groups"] = float(sum(1 for group in data if group and all(_sample_is_loss_masked(sample) for sample in group)))
        metrics["tg_opd/masked_group_rate"] = metrics["tg_opd/masked_groups"] / num_groups
        _accumulate_teacher_signal_metrics(metrics, data)
        _finalize_mopd_rates(metrics)
        metrics["tg_opd/probe_completed_rate"] = metrics.get("tg_opd/probe_completed", 0.0) / num_groups
        metrics["tg_opd/probe_completed_by_batch_end_rate"] = metrics["tg_opd/probe_completed_rate"]
        metrics["tg_opd/probe_timeout_rate"] = metrics.get("tg_opd/probe_timeout", 0.0) / num_groups
        metrics["tg_opd/probe_errored_rate"] = metrics.get("tg_opd/probe_errored", 0.0) / num_groups
        if metrics.get("tg_opd/probe_completed", 0.0) > 0:
            metrics["tg_opd/fallback_among_completed_rate"] = (
                metrics.get("tg_opd/fallback_groups", 0.0) / metrics["tg_opd/probe_completed"]
            )
        if args.tg_opd_mode in {"tg_opd", "offline_tg_opd"} and metrics["tg_opd/opd_rate"] == 0:
            logger.warning(
                "All TG-OPD prompts fell back to GRPO. Check --tg-opd-probe-pass-threshold "
                "and the teacher reward signal."
            )

        _emit_timing_metrics(accepted_timings, metrics)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)
    finally:
        if not pbar.disable:
            pbar.close()
        if student_state.pendings and not student_state.aborted:
            await sglang_rollout.abort(args, rollout_id)
        student_state.reset()

def _emit_timing_metrics(timings: list[dict[str, float]], metrics: dict[str, float]) -> None:
    """Compute per-rollout timing distribution and timeline metrics from the
    per-prompt timing dicts collected by :func:`_run_one_prompt`.

    Distribution metrics (mean/median/p95/max) cover what each prompt spent in
    the student vs teacher_logprob stages; timeline metrics describe how those
    stages overlapped within the rollout wall-clock so we can quantify the
    pipeline gap (e.g. ``teacher_logprob_after_student_s`` is the wall-clock
    that elapsed AFTER the last student prompt returned, attributable purely
    to draining teacher_logprob requests).
    """
    if not timings:
        return

    def _stats(values: list[float], key_prefix: str) -> None:
        if not values:
            return
        values = sorted(values)
        n = len(values)
        metrics[f"{key_prefix}/mean"] = sum(values) / n
        metrics[f"{key_prefix}/median"] = statistics.median(values)
        # Inclusive p95: lower index of the 95th percentile bucket
        p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
        metrics[f"{key_prefix}/p95"] = values[p95_idx]
        metrics[f"{key_prefix}/max"] = values[-1]
        metrics[f"{key_prefix}/min"] = values[0]

    student_walls = [t["student_wall"] for t in timings]
    logprob_walls = [t["logprob_wall"] for t in timings]
    prompt_totals = [t["prompt_total"] for t in timings]
    probe_latencies = [t["probe_latency"] for t in timings if "probe_latency" in t]
    probe_finish_flags = [
        t["probe_finish_before_student"] for t in timings if "probe_finish_before_student" in t
    ]
    probe_finish_batch_flags = [
        t["probe_finish_before_batch_end"] for t in timings if "probe_finish_before_batch_end" in t
    ]

    _stats(student_walls, "perf/tg_opd/timing/student_wall_s")
    _stats(logprob_walls, "perf/tg_opd/timing/logprob_wall_s")
    _stats(prompt_totals, "perf/tg_opd/timing/prompt_total_s")
    _stats(probe_latencies, "perf/tg_opd/probe_latency_s")
    if probe_finish_flags:
        metrics["perf/tg_opd/probe_finish_before_student_rate"] = (
            sum(probe_finish_flags) / len(probe_finish_flags)
        )
    if probe_finish_batch_flags:
        metrics["perf/tg_opd/probe_finish_before_batch_end_rate"] = (
            sum(probe_finish_batch_flags) / len(probe_finish_batch_flags)
        )

    # Rollout-wide timeline (offsets from the first prompt's start)
    rollout_origin = min(t["t_start"] for t in timings)
    last_student_end = max(t["t_student_end"] for t in timings) - rollout_origin
    first_logprob_start = min(t["t_logprob_start"] for t in timings) - rollout_origin
    last_logprob_start = max(t["t_logprob_start"] for t in timings) - rollout_origin
    last_logprob_end = max(t["t_end"] for t in timings) - rollout_origin

    metrics["perf/tg_opd/timeline/last_student_end_s"] = last_student_end
    metrics["perf/tg_opd/timeline/first_logprob_start_s"] = first_logprob_start
    metrics["perf/tg_opd/timeline/last_logprob_start_s"] = last_logprob_start
    metrics["perf/tg_opd/timeline/last_logprob_end_s"] = last_logprob_end
    # Wall-clock the rollout "spent only on teacher_logprob" — anything between
    # the last student returning and the last logprob completing.
    metrics["perf/tg_opd/timeline/logprob_after_student_s"] = max(
        0.0, last_logprob_end - last_student_end
    )
    # Overlap window where student and logprob were both in flight at some point.
    metrics["perf/tg_opd/timeline/student_logprob_overlap_s"] = max(
        0.0, last_student_end - first_logprob_start
    )


# ---------------------------------------------------------------------------
# Reward post-processing
# ---------------------------------------------------------------------------


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs):
    """Reward post-processing for mixed OPD/GRPO batches.

    For OPD samples (opd_mask=1) the task reward is zeroed because the OPD KL
    term provides the learning signal. For GRPO fallback samples (opd_mask=0)
    the reward is kept and group-normalized per group_index. If
    --tg-opd-unreliable-action=mask zeroed a sample loss_mask, its reward is
    also forced to zero and it is excluded from GRPO normalization. Group size
    is ``tg_opd_select_k`` (not ``n_samples_per_prompt``).
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = [
        0.0 if sample.opd_mask or _sample_is_loss_masked(sample) else raw_reward
        for sample, raw_reward in zip(samples, raw_rewards, strict=True)
    ]

    if (
        args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for pos, sample in enumerate(samples):
            if sample.opd_mask or _sample_is_loss_masked(sample):
                continue
            grouped[sample.group_index].append((pos, raw_rewards[pos]))

        use_std = (
            args.advantage_estimator in ["grpo", "gspo", "cispo"]
            and getattr(args, "grpo_std_normalization", True)
        )
        for indexed in grouped.values():
            values = torch.tensor([reward for _, reward in indexed], dtype=torch.float)
            values = values - values.mean()
            if use_std and values.numel() > 1:
                values = values / (values.std() + 1e-6)
            for (pos, _), reward in zip(indexed, values.tolist(), strict=True):
                rewards[pos] = reward

    return raw_rewards, rewards


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        # Evaluation has no TG-OPD-specific shape; defer to the stock path.
        output, _ = run(eval_rollout(args, rollout_id))
        return output
    return run(_generate_rollout_async(args, rollout_id, data_source))
