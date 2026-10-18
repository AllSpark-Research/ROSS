# RL/MOPD training data

| Domain | File | Records | Reward task |
|---|---|---:|---|
| Math | `math/train.jsonl` | 3,000 | DAPO-style final-answer checking |
| CodeIO | `codeio/train.jsonl` | 11,980 | 5,991 output prediction + 5,989 execution-based input prediction |
| IF | `if/train.jsonl` | 10,000 | Rule-based instruction following |
| Total | | 24,980 | |

These training splits are derived from public datasets. They contain
prompts and verifier labels/specifications, not the student rollout responses or
ROSS SFT annotations. Existing train split membership and row order are retained.
`metadata.domain` identifies the domain for teacher routing.

## Schema

Each record contains `prompt`, `label`, `data_source` and `metadata.reward_spec`.
Prompts are chat-message lists. `reward_spec` includes the source-specific ground
truth and/or `extra_info` required by the bundled verifier. In particular, IF
instruction IDs and kwargs and CodeIO reference programs remain present.

## Mixture

```bash
python data/prepare_rl_mopd.py --output artifacts/rl_mopd/train.jsonl
```

This concatenates the three files and shuffles using seed 42 without oversampling.
Use `--domains math` (or `codeio`, `if`) for a single-domain export. Apply
tokenizer-dependent length filtering separately for the target model.

## Sources and terms

The splits are compiled from public datasets, rather than company-internal data.
Source metadata identifies DAPO-style math, CodeIO examples derived from
`pyedur_full`, and `nano-v3-RL-instruction_following` / `IF-RL` records.

The split files do not include the exact upstream version citations or dataset
license files. Redistribution permission for these specific versions is therefore
unverified. Dataset use and redistribution are subject to the upstream terms;
the repository's Apache-2.0 code license does not cover these datasets.

`statistics.json` contains record counts, file sizes, and SHA-256 hashes.
