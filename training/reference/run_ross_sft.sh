#!/usr/bin/env bash
# Historical single-turn ROSS configuration, submitted to an existing Ray cluster.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SLIME_DIR=${SLIME_DIR:-$ROOT/training/framework}
for name in MEGATRON_DIR MODEL_CONFIG HF_CHECKPOINT REF_LOAD PROMPT_DATA OUTPUT_DIR; do
  [[ -n "${!name:-}" && "${!name}" == /* ]] || { echo "Set $name to an absolute path" >&2; exit 2; }
done
for name in SLIME_DIR MEGATRON_DIR HF_CHECKPOINT REF_LOAD; do
  [[ -d "${!name}" ]] || { echo "Missing directory: $name" >&2; exit 2; }
done
[[ -f "$MODEL_CONFIG" && -f "$PROMPT_DATA" ]] || { echo "Missing model configuration or annotated data" >&2; exit 2; }
declare -a MODEL_ARGS=() EXTRA_ARGS=()
source "$MODEL_CONFIG"
[[ ${#MODEL_ARGS[@]} -gt 0 ]] || { echo "MODEL_CONFIG must define MODEL_ARGS" >&2; exit 2; }
export PYTHONPATH="$ROOT/training/framework:$SLIME_DIR:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
CMD=(python3 "$SLIME_DIR/train_async.py"
  --actor-num-nodes "${ACTOR_NUM_NODES:-4}" --actor-num-gpus-per-node "${GPUS_PER_NODE:-8}"
  "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_CHECKPOINT" --ref-load "$REF_LOAD"
  --save "$OUTPUT_DIR" --save-interval "${SAVE_INTERVAL:-243}"
  --rollout-function-path slime_plugins.ross_sft.rollout.generate_rollout
  --prompt-data "$PROMPT_DATA" --input-key messages --rollout-shuffle
  --num-epoch "${NUM_EPOCH:-3}" --n-samples-per-prompt 1
  --rollout-batch-size "${GLOBAL_BATCH_SIZE:-128}" --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
  --rollout-max-context-len "${CUTOFF_LEN:-32768}"
  --loss-type sft_loss --loss-mask-type qwen3_5 --calculate-per-token-loss
  --disable-compute-advantages-and-returns --debug-train-only
  --optimizer adam --lr "${LR:-4e-6}" --lr-decay-style cosine --min-lr "${MIN_LR:-2e-7}"
  --lr-warmup-fraction 0.05 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --use-distributed-optimizer --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer --tensor-model-parallel-size "${TP_SIZE:-2}" --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE:-2}" --context-parallel-size "${CP_SIZE:-1}"
  --expert-model-parallel-size "${EP_SIZE:-2}" --expert-tensor-parallel-size "${ETP_SIZE:-1}"
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-10240}"
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash "${EXTRA_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
mkdir -p "$OUTPUT_DIR"
RUNTIME_ENV_JSON=$(python3 -c 'import json,os; print(json.dumps({"env_vars":{k:os.environ[k] for k in ["PYTHONPATH","CUDA_DEVICE_MAX_CONNECTIONS"]}}))')
cd "$SLIME_DIR"
exec ray job submit --address="${RAY_ADDRESS:-http://localhost:8265}" --runtime-env-json="$RUNTIME_ENV_JSON" -- "${CMD[@]}"
