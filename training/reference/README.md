# Experiment training references

These presets provide the RL, MOPD, and ROSS experiment configurations.
Set up a Ray cluster and convert model checkpoints before launching; the scripts
submit training jobs to the existing cluster.

| Script | Experiment configuration | Initialization/data supplied by caller |
| --- | --- | --- |
| `mopd_35b_three_domain.sh` | 35B three-domain MOPD; 400 rollouts; save every 50; 10,240 response tokens; selected-OPD; 32 actor + 24 rollout/teacher GPUs | Student HF and distributed checkpoints; teacher YAML; prepared three-domain mixture |
| `rl_4b_math.sh` | 4B Math RL; 120 rollouts; save every 24; 16,384 response tokens; 16 actor + 8 rollout GPUs | Base checkpoints; released Math data is selected by default |
| `ross_35b_base_first200.sh` | ROSS SFT on annotations from the first 200 upstream rollouts; 3 epochs; 31,228 historical rows; cutoff 32,768 | `BASE_REF_LOAD`; annotated SFT JSONL |
| `ross_35b_mopd199_first200.sh` | Same annotated data and SFT settings; initialize from MOPD iteration 199 | `MOPD_REF_LOAD`; annotated SFT JSONL |

The MOPD actor parallelism is TP/PP/CP/EP/ETP = 2/1/4/8/1, as in its original
shared runner. The 4B Math RL preset uses 2/1/2/1/1. Their batches are 64 prompts x 4
responses / 2 optimizer steps and 128 prompts x 8 responses / 4 optimizer steps,
respectively. All use GRPO advantages without group standard-deviation scaling,
PPO clipping at 0.2 and hard-rejection TIS. The MOPD preset enables routing
replay. The model-specific files also enable trajectory dumps.

## RL and MOPD usage

From the repository root, configure absolute paths on all Ray nodes:

```bash
export HF_CHECKPOINT=/path/to/student_hf
export REF_LOAD=/path/to/student_distributed
export MEGATRON_DIR=/path/to/Megatron-LM
export OUTPUT_DIR=/path/to/output
DRY_RUN=1 bash training/reference/rl_4b_math.sh
# Remove DRY_RUN after reviewing the generated command and cluster setup.
```

For MOPD, first run `data/prepare_rl_mopd.py` and set `PROMPT_DATA` to its output.
Set `SGLANG_CONFIG` to a local copy of `training/configs/mopd.example.yaml` with
all teacher paths configured. Run `mopd_35b_three_domain.sh`. The existing
TGOPD switch remains available through `TG_OPD_MODE`; see `../RL_MOPD.md`.
Use `DUMP_DETAILS_DIR` to choose a rollout dump location; its default is under
OUTPUT_DIR.

## ROSS SFT usage

`../run_masked_sft.sh` is a generic external-trainer template.
`run_ross_sft.sh` uses the opt-in
`slime_plugins.ross_sft.rollout.generate_rollout` adapter. It imports the original
`mask_utils_v6.py` implementation, without replacing Slime's ordinary SFT loader.

```bash
export HF_CHECKPOINT=/path/to/compatible_hf_model_and_tokenizer
export MOPD_REF_LOAD=/path/to/mopd_iter199_distributed
export MEGATRON_DIR=/path/to/Megatron-LM
export PROMPT_DATA=/path/to/annotated_first200.jsonl
export OUTPUT_DIR=/path/to/ross_sft_output
DRY_RUN=1 bash training/reference/ross_35b_mopd199_first200.sh
```

For the Base comparison, set BASE_REF_LOAD and use `ross_35b_base_first200.sh`.
The HF tokenizer/config must match the model architecture and token IDs of the
distributed initialization. The reference loader starts from REF_LOAD; resume
requires an explicit `--load` in the model configuration's EXTRA_ARGS.

The supplied RL prompt datasets are **not** annotated SFT datasets. Generate
annotations and export `messages` JSONL before SFT. The historical annotated
sets and checkpoints are not included. Prefilter records with the exact model
tokenizer; oversized records and records with no supervised tokens are rejected.
Recompute SAVE_INTERVAL if the usable row count changes: historical values are
243 for both 35B presets, with batch size 128.

This adapter supports the historical plain-text Qwen3.5-style `sft_char_spans`
path only. Missing/None spans retain ordinary assistant supervision; an empty
list excludes that assistant turn. The historical mask uses any token overlap
with selected characters and preserves original token IDs. `sft_region_exclude`
and multimodal content are unsupported. Agentic exports require the separate trainer
integration described in `../README.md`.
