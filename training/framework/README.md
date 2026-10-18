# Slime training framework for ROSS

This directory contains the Slime runtime packages, training entry points,
model definitions and conversion utilities used for ROSS RL/MOPD.
See [RL/MOPD instructions](../RL_MOPD.md) for setup, data, launch commands,
supported training modes and model requirements.

Public upstream: https://github.com/THUDM/slime

## Training features

- selected-OPD, online TGOPD and offline TGOPD rollout modes;
- configurable loss variants and hard-rejection TIS;
- three-domain reward adapters for Math, CodeIO, and instruction following;
- nested JSON input parsing in the Code input-execution verifier;
- a single-turn ROSS span-mask training adapter.

Teacher reliability gating is opt-in; ordinary MOPD uses selected-OPD by default.

The upstream [Apache-2.0 license](LICENSE) and source attribution notices are
retained in this directory.
