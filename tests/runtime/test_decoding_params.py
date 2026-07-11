"""Regression tests: the decoding-steer knobs that were previously stored on the
runner trainers but never reached the algorithm are now wired end-to-end.

- NudgingConfig.nudge_strength / safe_tokens / unsafe_tokens
- SafeDecodingConfig.banned_tokens

Each test drives the LogitsProcessor.combine() directly with synthetic logits so
it needs no model, and asserts the param actually changes the output logits.
"""
import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")


def _nudging(**cfg):
    import safetune.steer as S
    return S.NudgingProcessor(guide=object(), tokenizer_target=object(),
                              config=S.NudgingConfig(**cfg))


def _confident():
    # token 0 dominant → base top-1 prob high → no hand-over (coeff 0 → out=target)
    s = torch.full((1, 10), -5.0)
    s[0, 0] = 10.0
    return s


class TestNudgingParams:
    def test_unsafe_tokens_masked_to_neg_inf(self):
        out = _nudging(unsafe_tokens=[3]).combine(_confident(), torch.zeros(1, 10),
                                                  torch.zeros(1, 4, dtype=torch.long))
        assert out[0, 3].item() == float("-inf")

    def test_safe_tokens_boosted(self):
        target = _confident()
        out = _nudging(safe_tokens=[2], safe_token_boost=5.0).combine(
            target.clone(), torch.zeros(1, 10), torch.zeros(1, 4, dtype=torch.long))
        assert out[0, 2].item() == pytest.approx(target[0, 2].item() + 5.0)

    def test_nudge_strength_scales_handover(self):
        # uniform target → base uncertain → hand-over active; guide peaks at tok 5
        flat = torch.zeros(1, 10)
        guide = torch.zeros(1, 10)
        guide[0, 5] = 8.0
        ids = torch.zeros(1, 4, dtype=torch.long)
        v0 = _nudging(nudge_strength=0.0).combine(flat.clone(), guide, ids)[0, 5].item()
        v1 = _nudging(nudge_strength=1.0).combine(flat.clone(), guide, ids)[0, 5].item()
        v2 = _nudging(nudge_strength=2.0).combine(flat.clone(), guide, ids)[0, 5].item()
        assert v0 == pytest.approx(0.0)    # strength 0 → target
        assert v1 == pytest.approx(8.0)    # strength 1 → full guide (authors' switch)
        assert v2 == pytest.approx(16.0)   # strength 2 → amplified

    def test_defaults_reproduce_hard_switch(self):
        # Backward compat: default config == authors' hard switch (full hand-over).
        flat = torch.zeros(1, 10)
        guide = torch.zeros(1, 10)
        guide[0, 5] = 8.0
        out = _nudging().combine(flat, guide, torch.zeros(1, 4, dtype=torch.long))
        assert out[0, 5].item() == pytest.approx(8.0)  # equals guide on hand-over


class TestSafeDecodingBannedTokens:
    def _proc(self, **cfg):
        import safetune.steer as S
        return S.SafeDecodingProcessor(guide=object(), tokenizer_target=object(),
                                       prompt_length=0,
                                       config=S.SafeDecodingConfig(**cfg))

    @pytest.mark.parametrize("seq_len", [1, 99])  # in-window and out-of-window
    def test_banned_tokens_masked_both_windows(self, seq_len):
        out = self._proc(banned_tokens=[4], first_m=5).combine(
            _confident(), torch.zeros(1, 10), torch.zeros(1, seq_len, dtype=torch.long))
        assert out[0, 4].item() == float("-inf")

    def test_no_banning_by_default_out_of_window(self):
        # Out of window with no banned_tokens → returns the raw target unchanged.
        target = _confident()
        out = self._proc(first_m=5).combine(
            target.clone(), torch.zeros(1, 10), torch.zeros(1, 99, dtype=torch.long))
        assert torch.equal(out, target)


class TestRunnerForwardsDecodingParams:
    def test_nudging_trainer_forwards_params(self):
        import torch.nn as nn
        from safetune.runner.steer import NudgingTrainer
        t = NudgingTrainer(nn.Linear(2, 2), object(),
                           nudge_strength=2.5, safe_tokens=[1], unsafe_tokens=[2])
        proc = t._do_calibrate(harmful=["x"], harmless=["y"])
        assert proc.config.nudge_strength == 2.5
        assert proc.config.safe_tokens == [1]
        assert proc.config.unsafe_tokens == [2]

    def test_safedecoding_trainer_forwards_banned_tokens(self):
        import torch.nn as nn
        from safetune.runner.steer import SafeDecodingTrainer
        # SafeDecoding needs a distinct safety expert on the HF path (passing the
        # target as its own guide made the blend an identity no-op) — supply one.
        t = SafeDecodingTrainer(nn.Linear(2, 2), object(),
                                expert_model=nn.Linear(2, 2), banned_tokens=[7, 8])
        proc = t._do_calibrate(harmful=["x"], harmless=["y"])
        assert proc.config.banned_tokens == [7, 8]

    def test_safedecoding_trainer_requires_expert(self):
        import pytest
        import torch.nn as nn
        from safetune.runner.steer import SafeDecodingTrainer
        t = SafeDecodingTrainer(nn.Linear(2, 2), object(), banned_tokens=[7, 8])
        with pytest.raises(ValueError, match="expert"):
            t._do_calibrate(harmful=["x"], harmless=["y"])
