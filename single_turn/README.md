# Single-turn rollout annotation

This directory implements the single-turn ROSS annotation pipeline used for
Math, Code, and instruction-following rollouts. For verifier-positive Math and
Code responses, an LLM judge identifies the spans that should remain supervised,
a second pass audits the reconstructed target, and deterministic quality gates
validate the final boundaries. Verifier-positive instruction-following responses
retain full-response supervision. Text outside selected spans remains in context
but receives no SFT loss.

The complete judge instructions are in `prompts/annotation.txt`,
`prompts/audit.txt`, and `prompts/revision.txt`. `annotate.py` is the final
annotator; its supporting modules are in `stage_*.py`.

## Input

`run.py` accepts JSONL records with the following fields:

```json
{
  "prompt": "...",
  "response": "...",
  "reward": {"score": 1},
  "metadata": {"domain": "math", "reward_spec": {"ground_truth": "..."}}
}
```

Supported domain values are `math`, `codeio`, and `if`. Dataset-specific reward
metadata can be adapted before invoking the runner.

## Run

```bash
export ROSS_ENDPOINT=http://localhost:8000/v1/chat/completions
export ROSS_MODEL=your-judge-model
# Optional for authenticated endpoints:
# export ROSS_API_KEY=...
# export ROSS_AUTH_HEADER=Authorization
# export ROSS_AUTH_SCHEME=Bearer

python single_turn/run.py \
  --input data/verifier_positive.jsonl \
  --output outputs/single_turn_annotations.jsonl \
  --workers 8
```

The output includes the raw judge decision, resolved character spans,
post-annotation checks, and the final training-eligibility decision. The root-level `token_alignment.py` module projects retained character spans
onto the original token IDs using the training checkpoint tokenizer.

## Build masked-SFT JSONL

After annotation, join accepted decisions with the unchanged source records:

```bash
python data/prepare_single_turn_sft.py \
  --source data/verifier_positive.jsonl \
  --annotations outputs/single_turn_annotations.jsonl \
  --output outputs/single_turn_sft.jsonl
```

The converter fails closed when an accepted span is missing, overlapping, out
of range, or attached to a response that differs from the annotated source.
Whole-response selections remain ordinary message records; partial selections
carry `sft_char_spans` on the final assistant message. See
`training/README.md` for the training-side mask contract and reference launch.
