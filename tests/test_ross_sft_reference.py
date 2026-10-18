"""CPU tests for the historical span mask and opt-in SFT adapter."""
import copy
from types import SimpleNamespace

import pytest
from slime.utils.types import Sample
from slime_plugins.ross_sft.mask_utils_v6 import MultiTurnLossMaskGenerator
from slime_plugins.ross_sft import rollout


class CharacterTokenizer:
    """A deterministic Qwen-shaped template with one token per character."""
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = ''.join('<|im_start|>' + m['role'] + '\n' + m['content'].strip() + '<|im_end|>\n' for m in messages)
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}


def messages(spans):
    return [{'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': '  abcd  ', 'sft_char_spans': spans}]


@pytest.mark.parametrize('spans,expected', [([[3, 5]], 'bc'), ([], ''), (None, 'abcd<|im_end|>\n')])
def test_span_mask_preserves_original_tokens(spans, expected):
    tokenizer = CharacterTokenizer()
    ids, mask = MultiTurnLossMaskGenerator(tokenizer, 'qwen3_5').get_loss_mask(messages(spans))
    assert ids == tokenizer.apply_chat_template(messages(spans), tokenize=True)
    assert ''.join(chr(t) for t, keep in zip(ids, mask) if keep) == expected


def run_sample(monkeypatch, content, cutoff=1000):
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(rollout, 'TOKENIZER', tokenizer)
    monkeypatch.setattr(rollout, 'PROCESSOR', object())
    monkeypatch.setattr(rollout, 'MASK_GENERATOR', MultiTurnLossMaskGenerator(tokenizer, 'qwen3_5'))
    monkeypatch.setattr(rollout, 'SAMPLE_PRINTED', True)
    sample = Sample(prompt=copy.deepcopy(content), metadata={})
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1,
                           loss_mask_type='qwen3_5', rollout_max_context_len=cutoff)
    data = SimpleNamespace(get_samples=lambda n: [[sample]])
    rollout.generate_rollout(args, 0, data)
    return sample


def test_rollout_uses_span_mask(monkeypatch):
    sample = run_sample(monkeypatch, messages([[3, 5]]))
    assert sum(sample.loss_mask) == 2
    assert len(sample.loss_mask) == sample.response_length
    assert ''.join(chr(t) for t, keep in zip(sample.tokens[-sample.response_length:], sample.loss_mask) if keep) == 'bc'


@pytest.mark.parametrize('kind', ['exclude', 'empty', 'too_long'])
def test_rollout_rejects_unsupported_or_empty_samples(monkeypatch, kind):
    content = messages([] if kind == 'empty' else [[3, 5]])
    if kind == 'exclude': content[-1]['sft_region_exclude'] = [[0, 1]]
    with pytest.raises(ValueError):
        run_sample(monkeypatch, content, cutoff=1 if kind == 'too_long' else 1000)
