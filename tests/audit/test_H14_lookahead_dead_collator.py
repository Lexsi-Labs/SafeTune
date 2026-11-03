"""Audit H14: core/data_compiler/lookahead.py carried a second, unused LookAheadCollator.

It prepended a run of token-id 0 at position 0 (not the paper's preview) and nothing in src/, tests/,
docs/ or examples/ referenced it; the trainer uses AnswerPreviewCollator. LookAheadConfig stays.
"""
import safetune.core.data_compiler as dc
from safetune.core.data_compiler import lookahead


def test_the_unused_collator_is_gone_and_the_config_stays():
    assert not hasattr(lookahead, "LookAheadCollator")
    assert not hasattr(dc, "LookAheadCollator")
    assert hasattr(dc, "LookAheadConfig")
