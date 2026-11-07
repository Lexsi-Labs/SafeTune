"""Audit S16: NudgingTrainer used the target as its own guide (an identity processor); it now needs a distinct aligned guide_model."""
import pytest
import torch

from safetune.runner import steer as R

from _steer_common import HARMFUL, HARMLESS, build, prompt_ids

KW = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)


def test_trainer_raises_without_a_guide_model():
    T, tok = build(seed=0)
    with pytest.raises(ValueError, match="guide_model"):
        R.NudgingTrainer(T, tok).calibrate(HARMFUL, HARMLESS)


def test_generate_differs_from_the_plain_target_and_the_guide_alone():
    T, tok = build(seed=0)
    G, _ = build(seed=1)
    ids = prompt_ids(tok, HARMFUL[0])
    # a threshold above 1 hands every step to the guide, so the effect is visible on a random model
    proc = R.NudgingTrainer(T, tok, guide_model=G, top_prob_thres=2.0).calibrate(HARMFUL, HARMLESS)[0]
    out = proc.generate(ids, **KW)
    assert not torch.equal(out, T.generate(ids, **KW)), "the processor changed nothing"
    # the target still runs the loop: every step takes the guide's logits, so the text is the guide's greedy text
    assert torch.equal(out[:, ids.shape[1]:], G.generate(ids, **KW)[:, ids.shape[1]:])


def test_a_confident_base_model_is_left_alone():
    T, tok = build(seed=0)
    G, _ = build(seed=1)
    ids = prompt_ids(tok, HARMFUL[0])
    proc = R.NudgingTrainer(T, tok, guide_model=G, top_prob_thres=0.0).calibrate(HARMFUL, HARMLESS)[0]
    assert torch.equal(proc.generate(ids, **KW), T.generate(ids, **KW))


def test_trainer_rejects_the_target_as_its_own_guide():
    T, tok = build(seed=0)
    with pytest.raises(ValueError, match="guide_model is the target"):
        R.NudgingTrainer(T, tok, guide_model=T).calibrate(HARMFUL, HARMLESS)
