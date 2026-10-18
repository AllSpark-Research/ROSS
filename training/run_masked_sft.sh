#!/usr/bin/env bash
# Reference Slime launcher for ROSS masked SFT.
# The installed trainer must implement the sft_char_spans contract described in
# training/README.md. All machine-specific paths are supplied by the caller.
set -euo pipefail

required=(SLIME_DIR MEGATRON_DIR MODEL_CONFIG HF_CHECKPOINT REF_LOAD PROMPT_DATA OUTPUT_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
done
for path_name in SLIME_DIR MEGATRON_DIR; do
  if [[ ! -d "${!path_name}" ]]; then
    echo "directory does not exist: ${path_name}=${!path_name}" >&2
    exit 2
  fi
done
if [[ ! -f "${MODEL_CONFIG}" || ! -f "${PROMPT_DATA}" ]]; then
  echo "MODEL_CONFIG and PROMPT_DATA must be existing files" >&2
  exit 2
fi

# MODEL_CONFIG must define MODEL_ARGS as a Bash array. It may additionally
# define EVAL_ARGS and EXTRA_ARGS arrays for the installed Slime version.
declare -a MODEL_ARGS=()
declare -a EVAL_ARGS=()
declare -a EXTRA_ARGS=()
# shellcheck source=/dev/null
source "${MODEL_CONFIG}"
if [[ ${#MODEL_ARGS[@]} -eq 0 ]]; then
  echo "MODEL_CONFIG did not define MODEL_ARGS" >&2
  exit 2
fi

NUM_EPOCH="${NUM_EPOCH:-3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
LR="${LR:-4e-6}"
MIN_LR="${MIN_LR:-2e-7}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-2}"
ETP_SIZE="${ETP_SIZE:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
INPUT_KEY="${INPUT_KEY:-messages}"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3_5}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
RANK="${RANK:-0}"
LOAD_PATH="${LOAD_PATH:-${OUTPUT_DIR}}"
mkdir -p "${OUTPUT_DIR}" "${LOAD_PATH}"

if [[ -z "${SAVE_INTERVAL:-}" ]]; then
  rows=$(awk 'NF {n++} END {print n+0}' "${PROMPT_DATA}")
  SAVE_INTERVAL=$(( rows / GLOBAL_BATCH_SIZE ))
  (( SAVE_INTERVAL > 0 )) || SAVE_INTERVAL=1
fi

if [[ "${RANK}" != "0" ]]; then
  echo "joining Ray cluster at ${MASTER_ADDR}:${RAY_PORT}"
  ray start --address="${MASTER_ADDR}:${RAY_PORT}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --block
  exit 0
fi

HEAD_NODE_IP=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR}'))")
ray start --head \
  --port="${RAY_PORT}" \
  --node-ip-address="${HEAD_NODE_IP}" \
  --num-gpus="${GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host=localhost \
  --dashboard-port="${RAY_DASHBOARD_PORT}"

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT}")
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
  [[ -n "${WANDB_API_KEY:-}" ]] && WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
fi

export MEGATRON_DIR MASTER_ADDR
RUNTIME_ENV_JSON=$(python3 -c 'import json,os; print(json.dumps({"env_vars":{"PYTHONPATH":os.environ["MEGATRON_DIR"]+":"+os.environ.get("PYTHONPATH", ""),"CUDA_DEVICE_MAX_CONNECTIONS":"1","MASTER_ADDR":os.environ["MASTER_ADDR"]}}))')

cd "${SLIME_DIR}"
ray job submit --address="http://localhost:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --ref-load "${REF_LOAD}" \
  --load "${LOAD_PATH}" \
  --save "${OUTPUT_DIR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --prompt-data "${PROMPT_DATA}" \
  --input-key "${INPUT_KEY}" \
  --rollout-shuffle \
  --num-epoch "${NUM_EPOCH}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --loss-type sft_loss \
  --loss-mask-type "${LOSS_MASK_TYPE}" \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --optimizer adam \
  --lr "${LR}" \
  --lr-decay-style cosine \
  --min-lr "${MIN_LR}" \
  --lr-warmup-fraction "${LR_WARMUP_FRACTION}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --adam-beta1 "${ADAM_BETA1}" \
  --adam-beta2 "${ADAM_BETA2}" \
  --use-distributed-optimizer \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size "${TP_SIZE}" \
  --sequence-parallel \
  --pipeline-model-parallel-size "${PP_SIZE}" \
  --context-parallel-size "${CP_SIZE}" \
  --expert-model-parallel-size "${EP_SIZE}" \
  --expert-tensor-parallel-size "${ETP_SIZE}" \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  "${WANDB_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
