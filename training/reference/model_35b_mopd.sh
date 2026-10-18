#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../framework/scripts/models/qwen3.5-35B-A3B.sh"
EXTRA_ARGS+=(--sequence-parallel --use-dynamic-batch-size --max-tokens-per-gpu 6144
  --micro-batch-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2
  --use-precision-aware-optimizer --kl-loss-coef 0.0 --entropy-coef 0.0 --balance-data
  --rollout-max-prompt-len 16384 --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
  --dump-details "${DUMP_DETAILS_DIR:-$OUTPUT_DIR/rollouts}")
EXTRA_ARGS+=(--log-probs-chunk-size 64 --use-routing-replay --use-rollout-routing-replay --use-slime-router --opd-grpo-metrics-interval 5 --tg-opd-probe-max-response-len 10240 --tg-opd-teacher-server-concurrency 64)
