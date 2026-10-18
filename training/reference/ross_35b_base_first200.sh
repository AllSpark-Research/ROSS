#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# Annotated SFT data must be supplied separately; raw RL prompts are not SFT data.
: "${BASE_REF_LOAD:?Set BASE_REF_LOAD to the matching distributed initialization checkpoint}"
export REF_LOAD="${BASE_REF_LOAD}"
export MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/training/framework/scripts/models/qwen3.5-35B-A3B.sh}"
export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}
export NUM_EPOCH=3 CUTOFF_LEN=32768 SAVE_INTERVAL=${SAVE_INTERVAL:-243}
export LR=4e-6 MIN_LR=2e-7 GLOBAL_BATCH_SIZE=128
export TP_SIZE=2 PP_SIZE=2 CP_SIZE=1 EP_SIZE=2 ETP_SIZE=1 MAX_TOKENS_PER_GPU=10240
exec bash "$ROOT/training/reference/run_ross_sft.sh"
