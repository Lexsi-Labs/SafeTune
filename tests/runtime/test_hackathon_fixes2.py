"""Hackathon fixes, part 2 (validation report). CPU only, no downloads: tokenizers
are built in memory, models are tiny random configs, vLLM is stubbed."""
import sys
import types

import pytest

from .test_hackathon_cohere_fixes import _word_tok


# ── 1. double BOS on vLLM text prompts and the suite's WildGuard judge ───────

class _Out:
    def __init__(self):
        self.outputs = [types.SimpleNamespace(text="ok")]


class _StubLLM:
    """Records what vLLM would tokenize, using the same tokenizer."""
    def __init__(self, tok):
        self.tok, self.seen = tok, []

    def generate(self, prompts, *_a, **_k):
        self.seen += [self.tok.encode(p) for p in prompts]  # vLLM adds special tokens
        return [_Out() for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", mod)


def _one_bos(tok, ids_list):
    assert ids_list
    for ids in ids_list:
        assert ids.count(tok.bos_token_id) == 1, tok.convert_ids_to_tokens(ids)


def test_strip_bos_only_when_the_tokenizer_adds_bos():
    from safetune._refusal_helpers import _strip_bos
    tok = _word_tok()
    assert _strip_bos(tok, ["<bos> <|user|> hi", "plain"]) == [" <|user|> hi", "plain"]


def test_steer_vllm_render_prompts_single_bos(fake_vllm):
    from safetune.steer.backends.run import render_prompts
    from safetune.steer.backends.vllm_eval import _render_prompts
    tok = _word_tok()
    p = ["how do i bake bread"]
    _one_bos(tok, [tok.encode(t) for t in render_prompts(tok, p, True, strip_bos=True)])
    _one_bos(tok, [tok.encode(t) for t in _render_prompts(tok, p, True)])
    # the hf path tokenizes with add_special_tokens=False: keeps the template's BOS
    _one_bos(tok, [tok.encode(t, add_special_tokens=False) for t in render_prompts(tok, p, True)])


@pytest.mark.parametrize("cls", ["vllm.VllmBackend", "vllm_lens.VllmSteeredBackend"])
def test_pipeline_vllm_backends_single_bos(cls, fake_vllm, monkeypatch):
    import importlib
    mod_name, name = cls.split(".")
    mod = importlib.import_module(f"safetune.core.eval.pipeline.backends.{mod_name}")
    B = getattr(mod, name)
    b = B.__new__(B)
    b.chat_template, b.system_prompt = True, None
    from safetune.core.eval.pipeline.backends.base import GenerationConfig
    b.config = GenerationConfig()
    tok = _word_tok()
    b._loaded_tokenizer, b._loaded_llm = tok, _StubLLM(tok)
    b._loaded_sampling = {}
    monkeypatch.setattr(b, "_ensure_loaded", lambda: None, raising=False)
    monkeypatch.setattr(b, "_build_steering_payload", lambda: {}, raising=False)
    b.generate(["how do i bake bread"])
    _one_bos(tok, b._loaded_llm.seen)


def test_generate_responses_vllm_single_bos(monkeypatch):
    import transformers
    from safetune.evaluate import generate as g
    tok = _word_tok()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok)
    seen = []
    monkeypatch.setattr(g, "_generate_vllm",
                        lambda _m, prompts, **_k: seen.extend(tok.encode(p) for p in prompts) or [])
    g.generate_responses("m", ["how do i bake bread"], backend="vllm")
    _one_bos(tok, seen)


def test_suite_wildguard_judge_single_bos(monkeypatch):
    import importlib
    ev = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")
    calls = {}
    monkeypatch.setattr(ev, "get_wildguard", lambda: (None, None))

    def _run(tok, model, formatted, max_new_tokens=32, add_special_tokens=True):
        calls["add_special_tokens"] = add_special_tokens
        assert all(f.startswith("<s>") for f in formatted)
        return ["response refusal: yes"] * len(formatted)

    monkeypatch.setattr(ev, "_run_judge_batch", _run)
    ev._build_judge_scorer("wildguard").score([ev._pack_pair("how do i bake bread", "no")])
    assert calls["add_special_tokens"] is False


# ── 2. C4: ReSta streams one tensor at a time ────────────────────────────────

def _tiny_llama(seed, dtype, tie=False):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=tie)).to(dtype)


def _old_resta(ft, base, aligned, alpha, param_filter, dare, drop, seed):
    """The pre-streaming algorithm: full fp32 safety vector, full DARE copy,
    full merged state dict, then load_state_dict."""
    import torch
    a_sd, b_sd = aligned.state_dict(), base.state_dict()
    sv = {k: a_sd[k].float() - b_sd[k].float() for k in a_sd
          if k in b_sd and (not param_filter or any(f in k for f in param_filter))}
    if dare:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        sv = {k: (v * (torch.rand(v.shape, generator=g) < 1 - drop) * (1 / (1 - drop)))
              for k, v in sv.items()}
    new = {k: ((v.float() + alpha * sv[k]).to(v.dtype) if k in sv else v)
           for k, v in ft.state_dict().items()}
    ft.load_state_dict(new, strict=False)
    return ft


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("dare", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_streaming_resta_matches_the_old_algorithm(dare, dtype, tie):
    import torch
    from safetune.recover import apply_resta
    dt = getattr(torch, dtype)
    base, aligned = _tiny_llama(0, dt, tie), _tiny_llama(1, dt, tie)
    # tied embed / lm_head: no filter, so both names carry a (DARE) delta
    kw = dict(alpha=0.7, param_filter=None if tie else ["mlp", "self_attn"])
    old = _old_resta(_tiny_llama(2, dt, tie), base, aligned, dare=dare, drop=0.3, seed=5, **kw)
    new = apply_resta(_tiny_llama(2, dt, tie), base=base, aligned=aligned, dare=dare,
                      dare_drop_rate=0.3, dare_seed=5, **kw)
    for (k, o), n in zip(old.state_dict().items(), new.state_dict().values()):
        assert o.dtype == n.dtype and torch.equal(o, n), k
    # device="cpu" computes the deltas on CPU: same result
    cpu = apply_resta(_tiny_llama(2, dt, tie), base=base, aligned=aligned, dare=dare,
                      dare_drop_rate=0.3, dare_seed=5, device="cpu", **kw)
    assert all(torch.equal(a, b) for a, b in zip(cpu.state_dict().values(),
                                                 new.state_dict().values()))


def test_resta_trainer_passes_device(monkeypatch):
    import safetune.recover as R
    from safetune.runner.recover import ReStaTrainer
    seen = []
    monkeypatch.setattr(R, "apply_resta", lambda m, **kw: seen.append(kw["device"]) or m)
    ReStaTrainer(object(), base_model=None, aligned_model=None).apply()
    ReStaTrainer(object(), base_model=None, aligned_model=None, device="cpu").apply()
    ReStaTrainer(object(), base_model=None, aligned_model=None).apply(device="cpu")
    assert seen == [None, "cpu", "cpu"]


def _peak_new_bytes(fn, *models):
    """Peak bytes held by tensors allocated during ``fn()`` (CPU dispatch count)."""
    import weakref
    from torch.utils._python_dispatch import TorchDispatchMode
    known = {t.untyped_storage().data_ptr() for m in models for t in m.state_dict().values()}
    live, stat = {}, {"cur": 0, "peak": 0}

    def _free(ptr):
        stat["cur"] -= live.pop(ptr, 0)

    class _Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else [out]):
                if not hasattr(t, "untyped_storage"):
                    continue
                ptr = t.untyped_storage().data_ptr()
                if ptr in known or ptr in live or ptr == 0:
                    continue
                live[ptr] = t.untyped_storage().nbytes()
                stat["cur"] += live[ptr]
                stat["peak"] = max(stat["peak"], stat["cur"])
                weakref.finalize(t, _free, ptr)
            return out

    with _Count():
        fn()
    return stat["peak"]


def test_streaming_resta_extra_memory_is_one_tensor_not_the_model():
    import torch
    from safetune.recover import apply_resta
    base, aligned, ft = (_tiny_llama(i, torch.float32) for i in range(3))
    largest = max(p.numel() * 4 for p in ft.parameters())
    model = sum(p.numel() * 4 for p in ft.parameters())
    assert model > 20 * largest
    peak = _peak_new_bytes(lambda: apply_resta(ft, base=base, aligned=aligned, dare=True,
                                               dare_seed=0), base, aligned, ft)
    assert peak <= 6 * largest, (peak, largest, model)
    # the old algorithm held whole-model copies: the counter sees them
    ft2 = _tiny_llama(2, torch.float32)
    old = _peak_new_bytes(lambda: _old_resta(ft2, base, aligned, 1.0, None, True, 0.3, 0),
                          base, aligned, ft2)
    assert old > model


# ── 3. C6: batched generation left-pads a right-padded tokenizer ─────────────

_PROMPTS = ["how do i bake bread", "what is the capital of france paris"]


def _spy_masks(model):
    masks = []
    model.register_forward_pre_hook(
        lambda m, args, kwargs: masks.append(kwargs.get("attention_mask")), with_kwargs=True)
    return masks


def _assert_left_padded_then_restored(tok, masks):
    assert tok.padding_side == "right"
    m = next(x for x in masks if x is not None)
    assert bool((m[:, -1] == 1).all()), m  # last column real tokens: left padding
    assert bool((m == 0).any())  # the batch really was padded


def test_transformers_backend_left_pads_a_passed_in_tokenizer():
    import torch
    from safetune.core.eval.pipeline.backends.base import GenerationConfig
    from safetune.core.eval.pipeline.backends.transformers import TransformersBackend
    tok, model = _word_tok(), _tiny_llama(0, torch.float32).eval()
    tok.padding_side = "right"
    masks = _spy_masks(model)
    TransformersBackend(model, tok, config=GenerationConfig(max_new_tokens=2, batch_size=2)) \
        .generate(_PROMPTS)
    _assert_left_padded_then_restored(tok, masks)


@pytest.mark.parametrize("probe", ["_refusal_rate", "_kl_when_ablated"])
def test_refusal_direction_probes_left_pad(probe):
    import torch
    import safetune.steer.refusal_direction as rd
    tok, model = _word_tok(), _tiny_llama(0, torch.float32).eval()
    tok.padding_side = "right"
    masks = _spy_masks(model)
    if probe == "_refusal_rate":
        rd._refusal_rate(model, tok, _PROMPTS, max_new_tokens=2)
    else:
        rd._kl_when_ablated(model, tok, torch.randn(64), _PROMPTS)
    _assert_left_padded_then_restored(tok, masks)


def test_suite_evaluate_left_pads_and_restores(monkeypatch):
    import torch
    import importlib
    ev = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")
    monkeypatch.setattr(ev, "_build_judge_scorer", lambda _n: types.SimpleNamespace(
        score=lambda pairs: [0.0] * len(pairs)))
    tok, model = _word_tok(), _tiny_llama(0, torch.float32).eval()
    tok.padding_side = "right"
    masks = _spy_masks(model)
    ev._compute_metrics(model, _PROMPTS, tokenizer=tok, batch_size=2, max_new_tokens=2)
    _assert_left_padded_then_restored(tok, masks)


def test_left_padding_restores_on_error():
    from safetune._refusal_helpers import _left_padding
    tok = _word_tok()
    tok.padding_side = "right"
    with pytest.raises(RuntimeError), _left_padding(tok):
        assert tok.padding_side == "left"
        raise RuntimeError
    assert tok.padding_side == "right"
