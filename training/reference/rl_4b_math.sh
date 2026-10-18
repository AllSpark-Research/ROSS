#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# Reference configuration: 4B math RL with rollout dumps.
export MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/training/reference/model_4b_rl.sh}"
export PROMPT_DATA="${PROMPT_DATA:-$ROOT/data/rl_mopd/math/train.jsonl}"
export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2} ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
export TP_SIZE=2 PP_SIZE=1 CP_SIZE=2 EP_SIZE=1 ETP_SIZE=1
export ROLLOUT_GPUS_PER_ENGINE=2
export TRAIN_STEPS=${TRAIN_STEPS:-120} SAVE_INTERVAL=${SAVE_INTERVAL:-24}
export ROLLOUT_BATCH_SIZE=128 N_SAMPLES_PER_PROMPT=8 GLOBAL_BATCH_SIZE=256 NUM_STEPS_PER_ROLLOUT=4
export MAX_RESPONSE_LEN=16384 SGLANG_MEM_FRACTION_STATIC=0.75 SGLANG_MAX_RUNNING_REQUESTS=128
export TG_OPD_MODE=off
exec bash "$ROOT/training/run_rl_mopd.sh" rl
