"""Command construction checks; no training services are started."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, mode, tg_mode):
    config = tmp_path / 'model.sh'
    config.write_text('MODEL_ARGS=(--num-layers 2)\n')
    data = tmp_path / 'data.jsonl'
    data.write_text('{}\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    env = dict(os.environ, HF_CHECKPOINT=str(tmp_path), REF_LOAD=str(tmp_path),
               MEGATRON_DIR=str(tmp_path), MODEL_CONFIG=str(config), PROMPT_DATA=str(data),
               OUTPUT_DIR=str(tmp_path / 'output'), DRY_RUN='1', TG_OPD_MODE=tg_mode,
               SGLANG_CONFIG=str(ROOT / 'training/configs/mopd.example.yaml'),
               TG_OPD_OFFLINE_MANIFEST=str(manifest))
    return subprocess.run(['bash', str(ROOT / 'training/run_rl_mopd.sh'), mode],
                          env=env, text=True, capture_output=True)


@pytest.mark.parametrize('mode,tg_mode', [('rl', 'off'), ('mopd', 'selected_opd'),
                                         ('mopd', 'tg_opd'), ('mopd', 'offline_tg_opd')])
def test_valid_modes(tmp_path, mode, tg_mode):
    result = launch(tmp_path, mode, tg_mode)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    assert args[args.index('--tg-opd-mode') + 1] == tg_mode
    assert '--policy-loss-type' not in args
    assert '--rollout-global-dataset' not in args
    assert '--custom-generate-function-path' not in args
    fn = args[args.index('--rollout-function-path') + 1]
    if mode == 'rl':
        assert fn == 'slime.rollout.sglang_rollout.generate_rollout'
        assert '--use-opd' not in args
    else:
        assert fn == 'slime.rollout.tg_opd_rollout.generate_rollout'
        assert '--use-opd' in args
        assert json.loads(args[args.index('--mopd-teacher-routing') + 1]) == {
            'math': 'math_teacher', 'codeio': 'codeio_teacher', 'if': 'if_teacher'}
    assert ('--tg-opd-offline-manifest' in args) == (tg_mode == 'offline_tg_opd')


@pytest.mark.parametrize('mode,tg_mode', [('rl', 'tg_opd'), ('mopd', 'off'), ('mopd', 'invalid')])
def test_invalid_mode_combinations(tmp_path, mode, tg_mode):
    assert launch(tmp_path, mode, tg_mode).returncode != 0
