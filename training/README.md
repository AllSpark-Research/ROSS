# Masked-SFT training reference

`run_masked_sft.sh` is a Slime-compatible launcher for the single-turn experiment
configuration. Configure checkpoint paths and model-specific Slime settings
through environment variables.

The launcher uses the reported single-turn defaults: three epochs, global batch
size 128, Adam with learning rate `4e-6`, cosine decay to `2e-7`, warmup fraction
`0.05`, weight decay `0.1`, and betas `(0.9, 0.98)`. Parallelism defaults to
TP/PP/CP/EP/ETP = `2/2/1/2/1`; change it for the available installation and
model layout.

## Loss-mask contract

The input JSONL uses a `messages` field. For an assistant message:

- missing `sft_char_spans` means ordinary full assistant supervision;
- `sft_char_spans: [[a, b], ...]` keeps loss only on the listed half-open
  character intervals of `message["content"]`;
- `sft_region_exclude: [[a, b], ...]`, emitted for agentic trajectories, removes
  loss on the listed intervals while preserving the rest of that assistant
  message.

The data loader must project character intervals onto the original tokenization
and intersect them with its ordinary assistant-token loss mask. It must not
concatenate selected text or retokenize an edited response. `token_alignment.py`
contains the reference token-alignment implementation. The public launcher
assumes this contract has been connected to the installed trainer's loss-mask
hook; training frameworks expose that hook differently.

## Launch

The model configuration must define `MODEL_ARGS` as a Bash array compatible
with the installed Slime checkout. `EVAL_ARGS` and `EXTRA_ARGS` are optional.

```bash
export SLIME_DIR="third_party/slime"
export MEGATRON_DIR="third_party/megatron"
export MODEL_CONFIG="config/model.sh"
export HF_CHECKPOINT="models/base_hf"
export REF_LOAD="models/base_dist"
export PROMPT_DATA="data/ross_sft.jsonl"
export OUTPUT_DIR="checkpoints/ross"

# Optional distributed settings:
export ACTOR_NUM_NODES=4
export GPUS_PER_NODE=8
export MASTER_ADDR="master-host"
export RANK=0

bash training/run_masked_sft.sh
```

On worker nodes, set the same variables and a nonzero `RANK`; the script joins
the Ray cluster started by rank zero. Experiment tracking is disabled unless
`WANDB_PROJECT` is set. Supply authentication credentials through the runtime
environment when needed.

## Base and upstream initialization

The same script covers both initialization choices used in the paper. For a
Base-initialized run, `HF_CHECKPOINT` and `REF_LOAD` point to the Base model and
its distributed conversion. For an RL- or MOPD-initialized run, they point to
the corresponding Upstream checkpoint and its distributed conversion. The
annotated JSONL and all optimization settings remain unchanged when comparing
these initializations. If only an HF checkpoint is available, convert it with
the conversion utility supplied by the installed training framework before
launching SFT.

Agentic trajectories use the same masked-SFT objective but typically require a
longer sequence limit and a separate schedule. Their post-annotation JSONL is
created by `agentic/export_masks.py`; configure sequence length and parallelism
through the model configuration and `EXTRA_ARGS` rather than hard-coding them
in the launcher.

## Concrete experiment presets

The [reference directory](reference/README.md) provides Base/MOPD-initialized
35B ROSS presets. Their dedicated `run_ross_sft.sh` uses
the bundled historical `sft_char_spans` adapter. The generic launcher described
above still depends on the caller's trainer integration. The historical adapter
does not implement the agentic `sft_region_exclude` contract.
