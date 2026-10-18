# RL and multi-teacher on-policy distillation

The bundled `framework/` is a curated Slime 0.3.0 source snapshot from the
training codebase. It includes synchronous/asynchronous training, Ray actors,
Megatron/FSDP backends, SGLang rollout and weight synchronization, model
configurations, conversion tools, and local verifiers for the released domains.

## Installation

Use a Linux CUDA environment with PyTorch, SGLang and Megatron-LM compatible
with the model architecture. From the repository root:

```bash
pip install -r requirements.txt
pip install -e training/framework
```

`framework/requirements.txt` supplies Python dependencies; it does not install
a complete CUDA/Megatron/SGLang training stack. 

## Prepare data

```bash
python data/prepare_rl_mopd.py --output artifacts/rl_mopd/train.jsonl
```

This combines 3,000 Math, 11,980 CodeIO and 10,000 IF records and shuffles with
seed 42. It does not rebalance or tokenizer-filter them. For single-domain RL,
use the corresponding `data/rl_mopd/<domain>/train.jsonl` directly.
See `data/rl_mopd/README.md` for the data schema, sources, and terms.

## Configure and launch

Start Ray on your allocated nodes, using an identical checkout and mounted paths
on every node. The reference launcher submits to that existing cluster and does
not manage the cluster lifecycle. Set absolute paths:

```bash
export HF_CHECKPOINT=/path/to/student_hf
export REF_LOAD=/path/to/student_torch_dist
export MEGATRON_DIR=/path/to/Megatron-LM
export MODEL_CONFIG="$PWD/training/framework/scripts/models/qwen3.5-35B-A3B.sh"
export PROMPT_DATA="$PWD/artifacts/rl_mopd/train.jsonl"
export OUTPUT_DIR=/path/to/output
export RAY_ADDRESS=http://localhost:8265

DRY_RUN=1 bash training/run_rl_mopd.sh rl
bash training/run_rl_mopd.sh rl
```

Select the model configuration that matches the checkpoint. Checkpoints and
teachers are not supplied. Conversion tools are under `framework/tools/`.
The default RL reference allocates 4 x 8 actor GPUs and 16 separate rollout GPUs.
Set actor, rollout and parallelism variables for your hardware. These are
multi-node reference defaults. The launcher starts
from REF_LOAD; resume requires an explicit `--load` in `EXTRA_ARGS`, avoiding
an accidental load from a non-checkpoint output directory.

For MOPD, copy `configs/mopd.example.yaml` to a local configuration and replace
all three teacher paths.
The student and teachers must use compatible tokenizers and identical token IDs.

```bash
export SGLANG_CONFIG=/path/to/local/mopd.yaml
DRY_RUN=1 bash training/run_rl_mopd.sh mopd
bash training/run_rl_mopd.sh mopd
```

The example uses 32 actor GPUs, 16 student rollout GPUs and 8 teacher GPUs:
CodeIO TP4, Math TP2, IF TP2. `metadata.domain` selects the frozen teacher.
The launcher uses the original selected-OPD/TG-OPD orchestration. In MOPD
mode, the default is `TG_OPD_MODE=selected_opd`: select student responses and
attach the domain teacher's log probabilities, without reliability probes.
Task rewards are logged but zeroed for OPD samples; reverse KL supplies their
learning signal. A failed teacher logprob request uses the configured fallback.

| Launcher | TG_OPD_MODE | Behavior |
| --- | --- | --- |
| `rl` | `off` (default) | Ordinary reward-based RL, no TGOPD |
| `mopd` | `selected_opd` (default) | Multi-teacher OPD, no reliability gate |
| `mopd` | `tg_opd` | Online teacher probes and reliability gate |
| `mopd` | `offline_tg_opd` | Reliability from an offline manifest |

```bash
TG_OPD_MODE=tg_opd TG_OPD_PROBE_N=3 TG_OPD_PROBE_MIN_PASS=2 \
  bash training/run_rl_mopd.sh mopd

TG_OPD_MODE=offline_tg_opd TG_OPD_PROBE_N=10 TG_OPD_PROBE_MIN_PASS=7 \
  TG_OPD_OFFLINE_MANIFEST=/path/to/local/manifest.json \
  bash training/run_rl_mopd.sh mopd
```

`TG_OPD_SELECT_K` defaults to 4, with `TG_OPD_SELECT_STRATEGY=high_low`.
The default student sample count is also 4, so all responses are retained.
When changing sample counts, keep select-k no greater than samples-per-prompt
and set global batch size to rollout batch size times select-k divided by
steps-per-rollout. `TG_OPD_UNRELIABLE_ACTION=grpo` keeps reward-based fallback;
set it to `mask` to exclude unreliable samples from training.

Online probes run concurrently with students. Teacher logprob is attached as
student groups complete. At batch end, completed failed probes trigger fallback;
still-pending or errored probes retain OPD. Offline unreliable groups skip
teacher logprob. These are the original implementation's semantics.

All original TGOPD CLI arguments are preserved in `framework/slime/utils/arguments.py`.
Additional options can be appended via `EXTRA_ARGS` in your local model config.
The offline manifest builder and validator are under `framework/scripts/`.
Create manifests against the released dataset and exactly the same prompt,
label, metadata, tokenizer and gate settings used for training. Regenerate the
manifest whenever these inputs change, since they determine its fingerprints.

## Verifiers and execution

`slime_plugins/rewards/domain_adapter.py` uses bundled `domain_verifiers` by
default. Math uses the original DAPO answer check; CodeIO separates output
matching from execution-based input prediction; IF uses instruction IDs,
kwargs and strict aggregation. Unknown data sources raise an error.

The input-prediction verifier executes reference Python in a subprocess with
a timeout; it is not an OS security sandbox. Run this workload in an isolated
worker environment without credentials. W&B is disabled in the launcher.

## Relationship to ROSS SFT

The annotation pipelines are in `single_turn/` and `agentic/`. The concrete
presets in `reference/README.md` provide an opt-in single-turn ROSS adapter
for `sft_char_spans`. It does not support the agentic `sft_region_exclude` field;
that field fails explicitly. Follow `training/README.md` and `token_alignment.py`
for the broader integration contract.

## Additional configuration

The launcher uses `ADVANTAGE_ESTIMATOR=grpo` by default, including MOPD.
This framework selects the policy loss through that argument. For CISPO,
set `ADVANTAGE_ESTIMATOR=cispo EPS_CLIP=1.0 EPS_CLIP_HIGH=4.0`.
The global rollout dataset is enabled by the framework default.
