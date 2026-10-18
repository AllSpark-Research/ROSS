"""Dry runs for concrete experiment presets; no Ray or GPU services."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,expected', [
    ('rl_4b_math.sh', {'--num-rollout':'120', '--save-interval':'24', '--rollout-max-response-len':'16384'}),
    ('mopd_35b_three_domain.sh', {'--num-rollout':'400', '--tg-opd-mode':'selected_opd', '--pipeline-model-parallel-size':'1', '--expert-model-parallel-size':'8'}),
    ('ross_35b_base_first200.sh', {'--num-epoch':'3', '--save-interval':'243', '--rollout-max-context-len':'32768'}),
    ('ross_35b_mopd199_first200.sh', {'--num-epoch':'3', '--global-batch-size':'128'}),
])
def test_preset(tmp_path, name, expected):
    data = tmp_path/'data.jsonl'; data.write_text('{}\n')
    env = dict(os.environ, HF_CHECKPOINT=str(tmp_path), REF_LOAD=str(tmp_path),
               BASE_REF_LOAD=str(tmp_path), MOPD_REF_LOAD=str(tmp_path),
               MEGATRON_DIR=str(tmp_path), PROMPT_DATA=str(data), OUTPUT_DIR=str(tmp_path/'output'),
               SGLANG_CONFIG=str(ROOT/'training/configs/mopd.example.yaml'), DRY_RUN='1')
    for key in ['MODEL_CONFIG','SLIME_DIR','TG_OPD_MODE']: env.pop(key, None)
    result = subprocess.run(['bash', str(ROOT/'training/reference'/name)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    for key,value in expected.items(): assert argv[argv.index(key)+1] == value
    assert not (tmp_path/'output').exists()
    if name.startswith('ross_'):
        assert argv[argv.index('--rollout-function-path')+1] == 'slime_plugins.ross_sft.rollout.generate_rollout'
        assert '--load' not in argv
    else:
        assert '--dump-details' in argv
