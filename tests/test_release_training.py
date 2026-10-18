"""CPU contract tests. These do not start Ray, SGLang, or GPU training."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'training/framework'
sys.path.insert(0, str(FRAMEWORK))
from slime.utils.types import Sample
from slime_plugins.rewards.domain_verifiers import default_compute_score
from slime_plugins.rewards import domain_adapter


class RewardContracts(unittest.TestCase):
    def test_math_correct_and_wrong(self):
        self.assertEqual(default_compute_score('math_dapo', r'Answer: \boxed{42}', '42')['score'], 1)
        self.assertEqual(default_compute_score('math_dapo', r'Answer: \boxed{41}', '42')['score'], -1)

    def test_code_output(self):
        self.assertEqual(default_compute_score('simulation__codeio', '{"output":{"answer":42}}', '"output":{"answer":42}')['score'], 1)
        self.assertEqual(default_compute_score('simulation__codeio', '{"output":{"answer":41}}', '"output":{"answer":42}')['score'], 0)

    def test_code_input_execution_with_trusted_fixture(self):
        gt = {"reference_code": "def main_solution(x): return x * 2", "entrypoint": "main_solution", "target_output": 6}
        good = default_compute_score('simulation__codeio_input_exec', '{"input":{"x":3}}', gt)
        bad = default_compute_score('simulation__codeio_input_exec', '{"input":{"x":4}}', gt)
        self.assertEqual(good['score'], 1)
        self.assertEqual(bad['score'], 0)

    def test_if_strict(self):
        spec = {'instruction_id_list': ['startend:quotation'], 'kwargs': [None], 'aggregation': 'strict'}
        self.assertEqual(default_compute_score('nemotron_if_rl', '"hello"', extra_info=spec)['score'], 1)
        self.assertEqual(default_compute_score('nemotron_if_rl', 'hello', extra_info=spec)['score'], 0)

    def test_adapter_bundled_reward_and_thread(self):
        sample = Sample(response=r'Answer: \boxed{42}', response_length=4, label='42',
                        metadata={'reward_spec': {'data_source': 'math_dapo', 'ground_truth': '42', 'solution_mode': 'response'}})
        with patch.dict('os.environ', {'ROSS_REWARDS_ROOT': ''}):
            result = asyncio.run(domain_adapter.custom_rm(None, sample))
        self.assertEqual(result['score'], 1)


if __name__ == '__main__': unittest.main()
