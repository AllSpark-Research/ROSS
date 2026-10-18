# Agentic trajectory annotation

This directory implements the ROSS pipeline for long-horizon coding-agent
trajectories. It preserves the complete interaction history as context and
selectively removes SFT loss from localized defects in assistant turns.

The release has four stages:

1. convert each trajectory into a verified `*.steps.json` export with stable
   assistant-step IDs, tool-call IDs, and matched tool observations;
2. run `prepare.py` to validate the serialization and construct bounded
   annotation windows;
3. run `run.py` to perform semantic review followed by independent character
   boundary localization and deterministic validation;
4. run `export_masks.py` to withhold incomplete or unresolved trajectories and
   attach accepted exclusion intervals to assistant messages.

## Prompts

The complete prompts used by the final pipeline are included in:

- `annotation/prompts/semantic.txt`: semantic review of narrative claims,
  tool calls, causal claims, omissions, and uncertainty;
- `annotation/prompts/localize.txt`: exact localization of confirmed defects;

## Step-export schema

Each `*.steps.json` file contains an `initial_context_raw` string, a `summary`
with a stable sample name, and an ordered `steps` list. Every step contains:

- `id` such as `G001`;
- `assistant_raw`, the exact serialized assistant generation;
- `shared_thought_and_text`;
- `assistant_tokens`;
- `observation_raw`;
- `calls`, with tool name, unique call ID, raw return, and a flag confirming
  that the return was matched to the original text.

The preprocessing code fails closed on ambiguous tool serialization, duplicate
call IDs, unmatched returns, or altered source text.

## Run

```bash
python agentic/prepare.py \
  --input data/step_exports \
  --output artifacts/prepared

export ROSS_ENDPOINT=http://localhost:8000/v1/chat/completions
export ROSS_MODEL=your-judge-model
# Optional authentication:
# export ROSS_API_KEY=...
# export ROSS_AUTH_HEADER=Authorization
# export ROSS_AUTH_SCHEME=Bearer

python agentic/run.py \
  --prepared artifacts/prepared \
  --root artifacts/annotations \
  --workers 16

python agentic/export_masks.py \
  --input data/messages.jsonl \
  --prepared artifacts/prepared \
  --annotations artifacts/annotations \
  --output outputs/agentic_masked.jsonl
```

`messages.jsonl` must contain `sample_id` and the original ordered `messages`.
The exporter emits only the messages field expected by SFT loaders. A trajectory
is withheld when semantic review or localization is incomplete or unresolved.

Character exclusions use the same serialized assistant representation supplied
in `assistant_raw`. The root-level `token_alignment.py` module projects these intervals onto the
original stored token IDs using the training checkpoint tokenizer.

The resulting JSONL follows the exclusion form of the common training contract
in `training/README.md`. Agentic experiments use the same masked-SFT objective
with their long-context sequence limit and separate schedule.
