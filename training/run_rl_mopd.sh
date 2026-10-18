#!/usr/bin/env bash
# Run on an existing Ray cluster with the same checkout on every node.
set -euo pipefail
MODE=${1:-rl}
if [[ "$MODE" != rl && "$MODE" != mopd ]]; then
  echo "Usage: bash training/run_rl_mopd.sh {rl|mopd}" >&2; exit 2
fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FRAMEWORK_DIR=${FRAMEWORK_DIR:-$ROOT/training/framework}
for name in HF_CHECKPOINT REF_LOAD MODEL_CONFIG PROMPT_DATA OUTPUT_DIR MEGATRON_DIR; do
  if [[ -z "${!name:-}" ]]; then echo "Set $name before launching" >&2; exit 2; fi
  if [[ "${!name}" != /* ]]; then echo "$name must be an absolute path" >&2; exit 2; fi
done
for name in HF_CHECKPOINT REF_LOAD MEGATRON_DIR; do
  [[ -d "${!name}" ]] || { echo "Missing directory: $name" >&2; exit 2; }
done
[[ -f "$MODEL_CONFIG" && -f "$PROMPT_DATA" ]] || { echo "MODEL_CONFIG/PROMPT_DATA missing" >&2; exit 2; }
declare -a MODEL_ARGS=() EXTRA_ARGS=()
source "$MODEL_CONFIG"
[[ ${#MODEL_ARGS[@]} -gt 0 ]] || { echo "MODEL_CONFIG must define MODEL_ARGS" >&2; exit 2; }
METHOD_ARGS=()
ROLLOUT_FUNCTION=slime.rollout.sglang_rollout.generate_rollout
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-16}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo}
if [[ "$MODE" == mopd ]]; then
  : "${SGLANG_CONFIG:?Set SGLANG_CONFIG to a configured teacher YAML}"
  [[ "$SGLANG_CONFIG" == /* && -f "$SGLANG_CONFIG" ]] || { echo "SGLANG_CONFIG must be an existing absolute path" >&2; exit 2; }
  ROLLOUT_NUM_GPUS=${MOPD_ROLLOUT_NUM_GPUS:-24}
  MOPD_TEACHER_ROUTING=${MOPD_TEACHER_ROUTING:-'{"math":"math_teacher","codeio":"codeio_teacher","if":"if_teacher"}'}
  TG_OPD_MODE=${TG_OPD_MODE:-selected_opd}
  case "$TG_OPD_MODE" in selected_opd|tg_opd|offline_tg_opd) ;; *) echo "For MOPD set TG_OPD_MODE=selected_opd|tg_opd|offline_tg_opd" >&2; exit 2 ;; esac
  ROLLOUT_FUNCTION=slime.rollout.tg_opd_rollout.generate_rollout
  METHOD_ARGS=(--tg-opd-mode "$TG_OPD_MODE"
    --tg-opd-select-k "${TG_OPD_SELECT_K:-4}"
    --tg-opd-select-strategy "${TG_OPD_SELECT_STRATEGY:-high_low}"
    --tg-opd-probe-n "${TG_OPD_PROBE_N:-3}"
    --tg-opd-probe-min-pass "${TG_OPD_PROBE_MIN_PASS:-2}"
    --tg-opd-probe-pass-threshold "${TG_OPD_PROBE_PASS_THRESHOLD:-1.0}"
    --tg-opd-unreliable-action "${TG_OPD_UNRELIABLE_ACTION:-grpo}"
    --use-opd --opd-type sglang --opd-kl-coef "${OPD_KL_COEF:-1.0}"
    --sglang-config "$SGLANG_CONFIG" --mopd-teacher-routing "$MOPD_TEACHER_ROUTING"
    --custom-reward-post-process-path slime.rollout.tg_opd_rollout.post_process_rewards)
  if [[ "$TG_OPD_MODE" == offline_tg_opd ]]; then
    : "${TG_OPD_OFFLINE_MANIFEST:?Set TG_OPD_OFFLINE_MANIFEST for offline TGOPD}"
    [[ -f "$TG_OPD_OFFLINE_MANIFEST" ]] || { echo "Offline manifest missing" >&2; exit 2; }
    METHOD_ARGS+=(--tg-opd-offline-manifest "$TG_OPD_OFFLINE_MANIFEST")
  fi
else
  [[ "${TG_OPD_MODE:-off}" == off ]] || { echo "TGOPD requires mopd mode" >&2; exit 2; }
  METHOD_ARGS=(--tg-opd-mode off)
fi
export PYTHONPATH="$FRAMEWORK_DIR:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ROSS_REWARD_MAX_CONCURRENCY=${ROSS_REWARD_MAX_CONCURRENCY:-128}
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json, os
keys = ["PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "ROSS_REWARD_MAX_CONCURRENCY"]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys}}))
ENV
)
cd "$FRAMEWORK_DIR"
CMD=(python3 train_async.py
  --actor-num-nodes "${ACTOR_NUM_NODES:-4}" --actor-num-gpus-per-node "${GPUS_PER_NODE:-8}"
  --rollout-num-gpus "$ROLLOUT_NUM_GPUS" --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE:-4}"
  "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD"
  --save "$OUTPUT_DIR" --save-interval "${SAVE_INTERVAL:-50}"
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}'
  --rollout-shuffle --rollout-function-path "$ROLLOUT_FUNCTION"
  --custom-rm-path slime_plugins.rewards.domain_adapter.custom_rm --reward-key score
  --num-rollout "${TRAIN_STEPS:-400}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-64}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}" --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-2}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN:-10240}" --rollout-temperature "${TEMPERATURE:-1.0}"
  --advantage-estimator "$ADVANTAGE_ESTIMATOR" --disable-grpo-std-normalization
  --eps-clip "${EPS_CLIP:-0.2}" --eps-clip-high "${EPS_CLIP_HIGH:-0.2}"
  --use-tis --tis-clip "${TIS_CLIP:-2.0}" --custom-tis-function-path slime.backends.megatron_utils.loss.hard_rejection_tis_function
  --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9 --adam-beta2 0.98
  --tensor-model-parallel-size "${TP_SIZE:-2}" --pipeline-model-parallel-size "${PP_SIZE:-2}"
  --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size "${EP_SIZE:-2}"
  --expert-tensor-parallel-size "${ETP_SIZE:-1}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-64}"
  "${METHOD_ARGS[@]}" "${EXTRA_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
mkdir -p "$OUTPUT_DIR"
exec ray job submit --address="${RAY_ADDRESS:-http://localhost:8265}" --runtime-env-json="$RUNTIME_ENV_JSON" -- "${CMD[@]}"
