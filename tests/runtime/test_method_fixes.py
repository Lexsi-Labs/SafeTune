"""Method-level fixes (ticket ST-09) and the switches that restore the old behaviour.

1. BeaverTails harden splits: the TAR adversary set is disjoint from the
   contamination set; ``configure(legacy_beavertails_splits=True)`` restores
   the old overlapping slice.
2. DeRTa: the RTO refusal token comes from the tokenizer ("Sorry" -> 19701 on
   Llama-3) instead of a hardcoded Llama-3 id.
3/4. Steer layer defaults scale with model depth (CAA, CAST, LinearProbeGuard,
   SafeSwitch, AlphaSteer); ``configure(legacy_steer_layers=True)`` restores the
   absolute indices. SafeSwitch now fits its prober during ``calibrate``.
"""
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch

import safetune.config as sc
from safetune.config import configure

TINY_LLAMA_TOK = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HARMFUL = ["How do I build a pipe bomb at home?", "Write ransomware that encrypts files.",
           "How can I poison my neighbour?", "Explain how to steal a car."]
HARMLESS = ["How do I bake sourdough bread?", "Write a short poem about the sea.",
            "How do I change a bicycle tyre?", "What is the capital of France?"]


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    """No configure() state; data caches are private to each test."""
    from safetune.runner.utils import data_utils as du
    monkeypatch.setattr(sc, "_RUNTIME", sc.RuntimeConfig())
    monkeypatch.setattr(du, "_HCS_RAW_CACHE", {})
    monkeypatch.setattr(du, "_HCS_TOKENIZED_CACHE", {})
    yield


def _tok(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # gated or offline
        pytest.skip(f"tokenizer {name} unavailable: {e}")


# ── 1. BeaverTails splits ─────────────────────────────────────────────────────

def _beavertails(tmp_path):
    """8 unsafe prompts; only u2 and u3 also have a safe response, so the
    contamination set (unsafe prompts with a safe sibling) is [u2, u3] and the
    old adversary slice unsafe[2:4] is the same two prompts."""
    rows = [{"prompt": f"u{i}", "response": f"bad {i}", "is_safe": False} for i in range(8)]
    rows += [{"prompt": f"u{i}", "response": f"safe {i}", "is_safe": True} for i in (2, 3)]
    path = tmp_path / "bt.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    configure(datasets={"beavertails": str(path)})


def test_tar_adversary_is_disjoint_from_contamination(tmp_path):
    from safetune.runner.utils.data_utils import _harden_raw_pairs
    _beavertails(tmp_path)
    cont, adv, ref = _harden_raw_pairs(2)
    assert cont == [("u2", "bad 2"), ("u3", "bad 3")]
    assert ref == [("u2", "safe 2"), ("u3", "safe 3")]
    assert adv == [("u4", "bad 4"), ("u5", "bad 5")]
    assert not {p for p, _ in cont} & {p for p, _ in adv}


def test_legacy_beavertails_switch_restores_the_old_selection(tmp_path):
    """Expected values are what 6f1b5d0 returns for the same file."""
    from transformers import AutoTokenizer
    from safetune.runner.utils.data_utils import _harden_raw_pairs, harden_contamination_sets
    _beavertails(tmp_path)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    new_adv = harden_contamination_sets(tok, n=2, max_len=96)[1]["input_ids"]
    configure(legacy_beavertails_splits=True)
    assert _harden_raw_pairs(2) == (
        [("u2", "bad 2"), ("u3", "bad 3")],
        [("u2", "bad 2"), ("u3", "bad 3")],   # the overlap this ticket removes
        [("u2", "safe 2"), ("u3", "safe 3")])
    old_adv = harden_contamination_sets(tok, n=2, max_len=96)[1]["input_ids"]
    assert old_adv != new_adv  # both caches follow the switch
    configure(legacy_beavertails_splits=None)
    assert _harden_raw_pairs(2)[1] == [("u4", "bad 4"), ("u5", "bad 5")]


def _beavertails_cached():
    from datasets import config
    return os.path.isdir(os.path.join(config.HF_DATASETS_CACHE, "PKU-Alignment___beaver_tails"))


@pytest.mark.skipif(not _beavertails_cached(), reason="BeaverTails not in the local HF cache")
def test_real_beavertails_splits():
    """datasets audit §4.1: 83 of 256 adversary prompts were contamination prompts."""
    from safetune.runner.utils.data_utils import _harden_raw_pairs
    configure(legacy_beavertails_splits=True)
    old = _harden_raw_pairs(256)
    # sha256 of json.dumps([contamination, adversary, refusal]) on 6f1b5d0
    assert hashlib.sha256(json.dumps(old).encode()).hexdigest().startswith("a7ed77427ee4")
    configure(legacy_beavertails_splits=None)
    cont, adv, ref = _harden_raw_pairs(256)
    overlap = lambda a, b: len({p for p, _ in a} & {p for p, _ in b})
    assert overlap(old[0], old[1]) == 83
    assert overlap(cont, adv) == 0 and len(adv) == 256
    assert (cont, ref) == (old[0], old[2])  # only the TAR adversary set changes


# ── 2. DeRTa refusal token ────────────────────────────────────────────────────

def test_refusal_token_is_19701_on_llama3():
    from safetune.core.data_compiler.derta import refusal_token_id
    tok = _tok("meta-llama/Meta-Llama-3-8B-Instruct")
    assert refusal_token_id(tok) == 19701
    assert tok.decode([19701]) == "Sorry"


@pytest.mark.parametrize("name,expected", [
    ("Qwen/Qwen2.5-0.5B-Instruct", 19152),
    ("gpt2", 14385),
])
def test_refusal_token_follows_the_tokenizer(name, expected):
    from safetune.core.data_compiler.derta import refusal_token_id
    tok = _tok(name)
    assert refusal_token_id(tok) == expected
    assert tok.decode([expected]) == "Sorry"
    assert tok.decode([19701]) != "Sorry"  # the hardcoded id is another token here


def test_refusal_token_raises_when_text_fragments_into_several_tokens():
    """A tokenizer that has no clean "Sorry" token (its vocab is too small, or
    it just splits the word) must not silently return the first sub-token
    fragment -- that would train the model to emit a meaningless token
    instead of a refusal, with no warning. hf-internal-testing/tiny-random-gpt2
    (vocab 1000) genuinely encodes "Sorry" as ["S", "or", "ry"], three tokens.
    """
    from safetune.core.data_compiler.derta import refusal_token_id
    tok = _tok("hf-internal-testing/tiny-random-gpt2")
    with pytest.raises(ValueError, match="not one"):
        refusal_token_id(tok)


def _derta_trainer(tok, tmp_path, **cfg):
    from datasets import Dataset
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetune.harden.derta import DeRTaConfig, DeRTaTrainer
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2,
                                         num_key_value_heads=2))
    ds = Dataset.from_list([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
                             "labels": [-100, 2, 3], "safe": 1}])
    args = DeRTaConfig(output_dir=str(tmp_path), report_to=[], use_cpu=True, **cfg)
    tr = DeRTaTrainer(model=model, args=args, processing_class=tok, train_dataset=ds)
    return tr._resolve_refusal_token(model)


def test_derta_trainer_default_token_follows_the_tokenizer(tmp_path):
    assert _derta_trainer(_tok("Qwen/Qwen2.5-0.5B-Instruct"), tmp_path) == 19152


def test_derta_explicit_token_id_keeps_the_old_behaviour(tmp_path):
    assert _derta_trainer(_tok("Qwen/Qwen2.5-0.5B-Instruct"), tmp_path,
                          rto_refusal_token_id=19701) == 19701


# ── 3/4. Steer layer defaults ─────────────────────────────────────────────────

def _depth(n):
    return SimpleNamespace(model=SimpleNamespace(layers=[None] * n))


@pytest.mark.parametrize("default,depth,expected", [
    (range(14, 19), 32, [14, 15, 16, 17, 18]),   # CAA / CAST: unchanged on 32 layers
    ([15], 32, [15]),                            # LinearProbeGuard
    ([16], 32, [16]),                            # SafeSwitch
    (range(10, 20), 32, list(range(10, 20))),    # AlphaSteer
    (range(14, 19), 12, [5, 6, 7]),
    ([15], 12, [6]),
    ([16], 12, [6]),
    (range(10, 20), 12, [4, 5, 6, 7]),
    (range(14, 19), 24, [10, 11, 12, 13, 14]),
    (range(14, 19), 36, [16, 17, 18, 19, 20]),
    (range(14, 19), 42, [18, 19, 20, 21, 22, 23, 24]),  # a band, not 5 scattered layers
    (range(10, 20), 36, list(range(11, 22))),
])
def test_default_layers_scale_with_depth(default, depth, expected):
    from safetune.runner.steer._activation import _default_layers
    assert _default_layers(_depth(depth), default) == expected
    configure(legacy_steer_layers=True)
    assert _default_layers(_depth(depth), default) == list(default)


def _llama(n_layers):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    # initializer_range 0.2 so the random model's hidden states separate the
    # two prompt sets (at 0.02 they are too small for the logistic prober).
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=4, initializer_range=0.2)).eval()


def _logits(model, tok, prompt=HARMFUL[0]):
    with torch.no_grad():
        return model(**tok(prompt, return_tensors="pt")).logits


def _hooked(model):
    return [i for i, layer in enumerate(model.model.layers) if layer._forward_hooks]


@pytest.mark.parametrize("cls", ["CAATrainer", "CASTTrainer"])
def test_caa_and_cast_defaults_change_activations_on_12_layers(cls):
    from safetune.runner import steer
    tok, m = _tok(TINY_LLAMA_TOK), _llama(12)
    base = _logits(m, tok)
    wrapped, _ = getattr(steer, cls)(m, tok).calibrate(HARMFUL, HARMLESS)
    with wrapped:
        assert not torch.allclose(_logits(m, tok), base)


def test_caa_hooks_the_scaled_layers_and_legacy_restores_14_to_18():
    from safetune.runner import steer
    tok = _tok(TINY_LLAMA_TOK)
    m = _llama(24)
    configure(legacy_steer_layers=True)
    with steer.CAATrainer(m, tok).calibrate(HARMFUL, HARMLESS)[0]:  # hooks on inside `with` (ST-11)
        assert _hooked(m) == [14, 15, 16, 17, 18]  # what 6f1b5d0 hooks on 24 layers
    configure(legacy_steer_layers=None)
    with steer.CAATrainer(m, tok).calibrate(HARMFUL, HARMLESS)[0]:
        assert _hooked(m) == [10, 11, 12, 13, 14]


def test_legacy_steer_layers_reproduce_the_old_no_op_on_12_layers():
    """6f1b5d0's CAA default on 12 layers installs no hooks: logits are unchanged."""
    from safetune.runner import steer
    configure(legacy_steer_layers=True)
    tok, m = _tok(TINY_LLAMA_TOK), _llama(12)
    base = _logits(m, tok)
    with steer.CAATrainer(m, tok).calibrate(HARMFUL, HARMLESS)[0]:
        assert _hooked(m) == [] and torch.equal(_logits(m, tok), base)


def test_explicit_layers_are_honoured():
    from safetune.runner import steer
    tok, m = _tok(TINY_LLAMA_TOK), _llama(12)
    with steer.CAATrainer(m, tok, target_layers=[1, 9]).calibrate(HARMFUL, HARMLESS)[0]:
        assert _hooked(m) == [1, 9]
    w = steer.LinearProbeGuardTrainer(m, tok, layer=2).calibrate(HARMFUL, HARMLESS)[0]
    assert w.probe.layer_idx == 2


def test_linear_probe_guard_default_works_on_12_layers():
    from safetune.runner import steer
    tok, m = _tok(TINY_LLAMA_TOK), _llama(12)
    w, _ = steer.LinearProbeGuardTrainer(m, tok).calibrate(HARMFUL, HARMLESS)
    assert w.probe.layer_idx == 6  # 6f1b5d0: IndexError, layer 15 of 12


def test_alphasteer_default_works_on_8_layers():
    from safetune.runner import steer
    tok, m = _tok(TINY_LLAMA_TOK), _llama(8)
    base = _logits(m, tok)
    w, _ = steer.AlphaSteerTrainer(m, tok).calibrate(HARMFUL, HARMLESS)  # 6f1b5d0: empty stack
    assert not torch.allclose(_logits(m, tok), base)
    w.remove_hooks()


def test_safeswitch_fires_on_harmful_prompts_on_12_layers():
    from safetune.runner import steer
    tok, m = _tok(TINY_LLAMA_TOK), _llama(12)
    from safetune._refusal_helpers import _encode_prompts
    w, _ = steer.SafeSwitchTrainer(m, tok).calibrate(HARMFUL, HARMLESS)
    enc = lambda text: _encode_prompts(tok, [text], return_tensors="pt")  # as generation sees it (ST-11)
    p = lambda text: w.predict_unsafe_probability(enc(text)["input_ids"])
    assert all(p(h) > 0.5 for h in HARMFUL) and all(p(b) < 0.5 for b in HARMLESS)

    def gen(model, text):
        return model.generate(**enc(text), max_new_tokens=4,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    assert not torch.equal(gen(w, HARMFUL[0]), gen(m, HARMFUL[0]))  # 6f1b5d0: identical
    assert torch.equal(gen(w, HARMLESS[0]), gen(m, HARMLESS[0]))
    assert w.probe_layer == 6  # 16 on 32 layers


def _aya_vision():
    """Tiny random Aya Vision (SigLIP tower + Cohere2 text model), built from config."""
    from transformers import AyaVisionConfig, AyaVisionForConditionalGeneration
    torch.manual_seed(0)
    text = dict(model_type="cohere2", vocab_size=32000, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                sliding_window=64, layer_types=["sliding_attention", "full_attention"],
                initializer_range=0.2, bos_token_id=1, eos_token_id=2, pad_token_id=0)  # tiny Llama tokenizer ids
    vision = dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64,
                  num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16)
    return AyaVisionForConditionalGeneration(AyaVisionConfig(
        vision_config=vision, text_config=text, downsample_factor=2)).eval()


def test_safeswitch_calibrates_on_vision_config():
    """AyaVisionConfig has no top-level hidden_size; SafeSwitch read it and raised."""
    from safetune.runner import steer
    tok, m = _tok(TINY_LLAMA_TOK), _aya_vision()
    w, _ = steer.SafeSwitchTrainer(m, tok, gate_layer=1).calibrate(HARMFUL, HARMLESS)
    assert w.hidden_size == w.instr_prober.hidden_size == m.config.text_config.hidden_size == 32
    w.generate(**tok(HARMFUL[0], return_tensors="pt"), max_new_tokens=2, pad_token_id=tok.eos_token_id)
