#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# Reference configuration: three-domain MOPD, 400 rollouts, no gate by default.
export MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/training/reference/model_35b_mopd.sh}"
: "${PROMPT_DATA:?Use the mixture produced by data/prepare_rl_mopd.py}"
export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4} MOPD_ROLLOUT_NUM_GPUS=${MOPD_ROLLOUT_NUM_GPUS:-24}
export TP_SIZE=2 PP_SIZE=1 CP_SIZE=4 EP_SIZE=8 ETP_SIZE=1
export TRAIN_STEPS=${TRAIN_STEPS:-400} SAVE_INTERVAL=${SAVE_INTERVAL:-50}
export ROLLOUT_BATCH_SIZE=64 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=128 NUM_STEPS_PER_ROLLOUT=2
export MAX_RESPONSE_LEN=10240 TG_OPD_MODE=${TG_OPD_MODE:-selected_opd}
export TG_OPD_SELECT_K=4 TG_OPD_PROBE_N=10 TG_OPD_PROBE_MIN_PASS=7
export SGLANG_MEM_FRACTION_STATIC=0.60 SGLANG_MAX_RUNNING_REQUESTS=64
exec bash "$ROOT/training/run_rl_mopd.sh" mopd
