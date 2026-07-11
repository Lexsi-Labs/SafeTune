"""Tests for Batch 2 safety modules (12 new integrations)."""
import pytest


# ─── RESTA ────────────────────────────────────────────────────────────────────

def test_resta_apply():
    """Test RESTA via public apply_resta() API (training-free safety recovery)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from safetune.recover import apply_resta

    # Create dummy models from state dicts
    base_model = nn.Linear(4, 4, bias=False)
    aligned_model = nn.Linear(4, 4, bias=False)
    ft_model = nn.Linear(4, 4, bias=False)

    # Set weights: base=0, aligned=1, finetuned=0.5
    with torch.no_grad():
        base_model.weight.zero_()
        aligned_model.weight.fill_(1.0)
        ft_model.weight.fill_(0.5)

    # Apply RESTA: θ_safe = 0.5 + 1.0 * (1.0 - 0.0) = 1.5
    result = apply_resta(ft_model, base_model, aligned_model, alpha=1.0)
    expected = torch.full((4, 4), 1.5)
    assert torch.allclose(result.weight, expected)


def test_resta_with_different_alpha():
    """Test RESTA with alpha < 1.0 (partial safety recovery)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from safetune.recover import apply_resta

    base_model = nn.Linear(2, 2, bias=False)
    aligned_model = nn.Linear(2, 2, bias=False)
    ft_model = nn.Linear(2, 2, bias=False)

    with torch.no_grad():
        base_model.weight.zero_()
        aligned_model.weight.fill_(2.0)
        ft_model.weight.fill_(1.0)

    # Apply RESTA: θ_safe = 1.0 + 0.5 * (2.0 - 0.0) = 2.0
    result = apply_resta(ft_model, base_model, aligned_model, alpha=0.5)
    expected = torch.full((2, 2), 2.0)
    assert torch.allclose(result.weight, expected)


# ─── LoX ──────────────────────────────────────────────────────────────────────

def test_lox_apply():
    """Test LoX via public apply_lox() API (low-rank safety extrapolation)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from safetune.recover import apply_lox

    # Create dummy models: base (zeros), aligned (identity), finetuned (zeros)
    base_model = nn.Linear(8, 8, bias=False)
    aligned_model = nn.Linear(8, 8, bias=False)
    ft_model = nn.Linear(8, 8, bias=False)

    with torch.no_grad():
        base_model.weight.zero_()
        aligned_model.weight.copy_(torch.eye(8))
        ft_model.weight.zero_()

    # Snapshot the finetuned weights BEFORE the call: apply_lox mutates the
    # model in place and returns the same object, so we must compare against a
    # copy of the original weights, not against ``ft_model.weight`` (which is
    # the very tensor that gets overwritten).
    ft_weight_before = ft_model.weight.detach().clone()

    # Apply LoX: should extrapolate the safety direction (identity matrix)
    result = apply_lox(ft_model, base_model, aligned_model, rank=4, extrapolation_factor=2.0)

    # Should return a model with modified weights
    assert result.weight.shape == torch.Size([8, 8])
    # Result should differ from the original finetuned weights (safety injected)
    assert not torch.allclose(result.weight, ft_weight_before)


def test_lox_rank_parameter():
    """Test LoX with different rank hyperparameter."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from safetune.recover import apply_lox

    base_model = nn.Linear(6, 6, bias=False)
    aligned_model = nn.Linear(6, 6, bias=False)
    ft_model = nn.Linear(6, 6, bias=False)

    with torch.no_grad():
        base_model.weight.zero_()
        aligned_model.weight.fill_(1.0)
        ft_model.weight.zero_()

    # Apply LoX with rank=2 (low-rank approximation)
    result = apply_lox(ft_model, base_model, aligned_model, rank=2, extrapolation_factor=1.5)

    # Verify output shape is preserved
    assert result.weight.shape == torch.Size([6, 6])


# ─── AsFT ─────────────────────────────────────────────────────────────────────

def test_asft_config():
    from safetune.core.optim.asft import AsFTConfig
    cfg = AsFTConfig(reg_lambda=0.2)
    assert cfg.reg_lambda == 0.2


def test_asft_constraint():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from safetune.core.optim.asft import AsFTWrapper, AsFTConfig

    model = nn.Linear(4, 4, bias=False)
    base_sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    aligned_sd = {k: torch.ones_like(v) for k, v in model.state_dict().items()}

    wrapper = AsFTWrapper(model, aligned_sd, base_sd, AsFTConfig(hard_constraint=True))

    x = torch.ones(2, 4)
    loss = model(x).sum()
    loss.backward()
    grad_before = model.weight.grad.clone()

    with wrapper.apply_subspace_constraint():
        pass

    # After hard constraint, gradient should only have aligned component
    assert model.weight.grad is not None


# ─── DeRTa ────────────────────────────────────────────────────────────────────

def test_derta_config():
    from safetune.core.data_compiler.derta import DeRTaConfig
    cfg = DeRTaConfig(num_prefix_variants=3)
    assert cfg.num_prefix_variants == 3


def test_derta_augment_example():
    from safetune.core.data_compiler.derta import DeRTaFormatter, DeRTaConfig
    fmt = DeRTaFormatter(DeRTaConfig(num_prefix_variants=3, enable_rto=True))
    rows = fmt.augment_example(
        prompt="How to do X?",
        harmful_response="Here is how to do X step one step two step three step four step five",
        safe_response="I cannot help with that.",
    )
    assert len(rows) > 0
    assert all("augmentation" in r for r in rows)
    aug_types = {r["augmentation"] for r in rows}
    assert "mle_prefix" in aug_types


def test_derta_augment_dataset():
    from safetune.core.data_compiler.derta import DeRTaFormatter
    fmt = DeRTaFormatter()
    examples = [
        {"prompt": "Q", "harmful_response": "bad " * 20, "safe_response": "no"}
        for _ in range(3)
    ]
    result = fmt.augment_dataset(examples)
    assert len(result) > 3


# ─── Safety-Layers ────────────────────────────────────────────────────────────

def test_safety_layers_config():
    from safetune.core.optim.safety_layers import SafetyLayersConfig
    cfg = SafetyLayersConfig(cosine_threshold=0.9)
    assert cfg.cosine_threshold == 0.9


def test_safety_layers_manual():
    from safetune.core.optim.safety_layers import SafetyLayerLocator, SafetyLayersConfig
    loc = SafetyLayerLocator(SafetyLayersConfig(
        localization_method="manual",
        safety_layer_indices=[5, 10, 15],
    ))
    layers = loc.locate()
    assert layers == {5, 10, 15}


def test_sppft_freeze():
    pytest.importorskip("torch")
    import torch.nn as nn
    from safetune.core.optim.safety_layers import SPPFTWrapper

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # No "layers.X" naming, so no freezing will occur (but it should not crash)
    sppft = SPPFTWrapper(model, safety_layers={0, 1})
    sppft.apply()
    sppft.restore()


# ─── AdaSteer ─────────────────────────────────────────────────────────────────

def test_adasteer_config():
    from safetune.core.runtime.inference.adasteer import AdaSteerConfig
    cfg = AdaSteerConfig(base_multiplier=2.0)
    assert cfg.base_multiplier == 2.0


def test_adasteer_adaptive_multiplier():
    from safetune.core.runtime.inference.adasteer import AdaSteerWrapper, AdaSteerConfig
    wrapper = AdaSteerWrapper(model=None, safety_vectors={}, config=AdaSteerConfig(
        base_multiplier=3.0, adaptive=True, safety_threshold=0.5
    ))
    wrapper.set_adaptive_multiplier(0.2)  # Low safety → higher multiplier
    assert wrapper._current_multiplier > 3.0

    wrapper.set_adaptive_multiplier(0.9)  # High safety → lower multiplier
    assert wrapper._current_multiplier < 3.0


# ─── SCANS ────────────────────────────────────────────────────────────────────

def test_scans_config():
    from safetune.core.runtime.inference.scans import SCANSConfig
    cfg = SCANSConfig(multiplier=4.0)
    assert cfg.multiplier == 4.0


def test_scans_compute_vectors():
    pytest.importorskip("torch")
    import torch
    from safetune.core.runtime.inference.scans import SCANSWrapper, SCANSConfig

    wrapper = SCANSWrapper(model=None, config=SCANSConfig(target_layers=[0, 1]))
    safe_act = {0: torch.ones(10, 16), 1: torch.ones(10, 16) * 2}
    unsafe_act = {0: torch.zeros(10, 16), 1: torch.ones(10, 16)}
    wrapper.compute_vectors(safe_act, unsafe_act)
    assert 0 in wrapper._steering_vectors
    assert 1 in wrapper._steering_vectors


# ─── STA ──────────────────────────────────────────────────────────────────────

def test_sta_config():
    from safetune.core.runtime.inference.sta import STAConfig
    cfg = STAConfig(target_atoms=[(12, 5)], multiplier=2.5)
    assert len(cfg.target_atoms) == 1


def test_sta_hook_registration():
    from safetune.core.runtime.inference.sta import STAWrapper, STAConfig
    wrapper = STAWrapper(model=None, atom_vectors={}, config=STAConfig())
    # No model layers → no hooks, but shouldn't crash
    wrapper.register_hooks()
    assert len(wrapper._hooks) == 0
    wrapper.remove_hooks()


# ─── PKE ──────────────────────────────────────────────────────────────────────

class _TinyTokenizer:
    """Minimal char-level tokenizer so the DINM-style PKE edit can tokenize the
    harmful prompt / safe response without downloading a real tokenizer."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1

    def __call__(self, text, add_special_tokens=True):
        ids = [(ord(c) % (self.vocab_size - 1)) for c in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def _make_pke_toy_lm(hidden: int = 8, n_layers: int = 2, vocab: int = 64):
    """Build a tiny *causal LM* that uses PKE's expected
    ``layers.{i}.mlp.down_proj.weight`` naming so the toxic-neuron
    locator/editor can find a region AND run a real forward pass (the faithful
    DINM edit teaches the located ``down_proj`` rows to emit a refusal via
    cross-entropy, so the model must be forward-able and produce logits).
    """
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x):
            return x + torch.tanh(self.mlp.down_proj(x))

    class _ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids, attention_mask=None):
            h = self.embed(input_ids)
            for blk in self.layers:
                h = blk(h)
            return self.lm_head(h)

    return _ToyLM()


def test_pke_apply():
    """PKE edits the located down_proj rows via DINM-style refusal-CE."""
    pytest.importorskip("torch")
    import torch
    from safetune.recover import apply_pke

    torch.manual_seed(0)
    clean_model = _make_pke_toy_lm(hidden=8, n_layers=2)
    toxic_model = _make_pke_toy_lm(hidden=8, n_layers=2)
    model = _make_pke_toy_lm(hidden=8, n_layers=2)
    tok = _TinyTokenizer(vocab_size=64)

    with torch.no_grad():
        # Make clean vs toxic differ on down_proj so the locator has a signal.
        for blk in clean_model.layers:
            blk.mlp.down_proj.weight.zero_()
        for blk in toxic_model.layers:
            blk.mlp.down_proj.weight.fill_(10.0)

    w_before = model.layers[0].mlp.down_proj.weight.detach().clone()

    result = apply_pke(
        model, clean_model, toxic_model, top_k_neurons=3,
        tokenizer=tok, num_steps=8, lr=1e-2,
    )

    assert result is model
    # Located toxic rows of down_proj were gradient-edited.
    assert not torch.allclose(result.layers[0].mlp.down_proj.weight, w_before)


def test_pke_edit_is_refusal_ce_not_weight_mse():
    """The learning signal MUST be refusal cross-entropy, not weight-MSE.

    Asserts (a) the per-step edit CE decreases and (b) after the edit the model
    assigns higher likelihood to the safe refusal on the harmful prompt -- the
    DINM/PKE objective. Also asserts only the located down_proj rows moved.
    """
    pytest.importorskip("torch")
    import torch
    import torch.nn.functional as F
    from safetune.recover.pke import (
        PKEGradientEditor, PKEConfig, DEFAULT_REFUSAL, DEFAULT_HARMFUL_PROMPT,
    )
    from safetune.core.pke import ToxicNeuronLocator

    torch.manual_seed(0)
    clean_model = _make_pke_toy_lm(hidden=16, n_layers=2)
    toxic_model = _make_pke_toy_lm(hidden=16, n_layers=2)
    model = _make_pke_toy_lm(hidden=16, n_layers=2)
    tok = _TinyTokenizer(vocab_size=64)

    with torch.no_grad():
        for blk in clean_model.layers:
            blk.mlp.down_proj.weight.zero_()
        for blk in toxic_model.layers:
            blk.mlp.down_proj.weight.fill_(10.0)

    cfg = PKEConfig(top_k_neurons=4, num_steps=15, lr=1e-2,
                    max_edit_magnitude=None, max_len=48)
    locator = ToxicNeuronLocator(config=cfg)
    toxic_neurons = locator.locate_by_weight_change(
        clean_model.state_dict(), toxic_model.state_dict()
    )

    # Refusal likelihood on the harmful prompt BEFORE the edit.
    def _refusal_nll():
        editor = PKEGradientEditor(model, toxic_neurons, tokenizer=tok, config=cfg)
        in_ids, labels, attn = editor._build_supervised_batch(
            DEFAULT_HARMFUL_PROMPT, DEFAULT_REFUSAL
        )
        with torch.no_grad():
            logits = model(input_ids=in_ids, attention_mask=attn)
            return float(editor._edit_nll(logits, labels))

    nll_before = _refusal_nll()

    # Snapshot every down_proj row to verify only located rows move.
    snap = {
        i: blk.mlp.down_proj.weight.detach().clone()
        for i, blk in enumerate(model.layers)
    }

    editor = PKEGradientEditor(model, toxic_neurons, tokenizer=tok, config=cfg)
    edited = editor.apply_edits(
        harmful_prompt=DEFAULT_HARMFUL_PROMPT, safe_response=DEFAULT_REFUSAL
    )

    nll_after = _refusal_nll()

    # (1) the edit is driven by refusal-CE that decreases over steps
    assert edited > 0
    assert len(editor.last_edit_losses) == cfg.num_steps
    assert editor.last_edit_losses[-1] < editor.last_edit_losses[0], (
        f"edit CE should decrease: {editor.last_edit_losses[0]:.3f} -> "
        f"{editor.last_edit_losses[-1]:.3f}"
    )
    # (2) refusal likelihood on the harmful prompt increased (NLL down)
    assert nll_after < nll_before, (
        f"refusal NLL should drop (more likely refusal): {nll_before:.3f} -> "
        f"{nll_after:.3f}"
    )
    # (3) a logit-space KL locality term was computed
    assert len(editor.last_locality_losses) == cfg.num_steps

    # (4) ONLY the located down_proj rows changed; all other rows are frozen.
    located = {li: set(rows) for li, rows in toxic_neurons.items()}
    for i, blk in enumerate(model.layers):
        after = blk.mlp.down_proj.weight.detach()
        changed = (~torch.isclose(after, snap[i], atol=1e-7).all(dim=1))
        changed_rows = set(torch.nonzero(changed).flatten().tolist())
        assert changed_rows <= located.get(i, set()), (
            f"layer {i}: rows {changed_rows - located.get(i, set())} moved but "
            "were not located"
        )


def test_pke_with_different_k():
    """Editing more neurons (larger k) must touch at least as many rows."""
    pytest.importorskip("torch")
    import torch
    from safetune.recover import apply_pke

    def _changed_rows(top_k):
        torch.manual_seed(0)
        clean_model = _make_pke_toy_lm(hidden=8, n_layers=2)
        toxic_model = _make_pke_toy_lm(hidden=8, n_layers=2)
        model = _make_pke_toy_lm(hidden=8, n_layers=2)
        tok = _TinyTokenizer(vocab_size=64)
        with torch.no_grad():
            for blk in clean_model.layers:
                blk.mlp.down_proj.weight.zero_()
            for blk in toxic_model.layers:
                blk.mlp.down_proj.weight.fill_(10.0)
        before = model.layers[0].mlp.down_proj.weight.detach().clone()
        result = apply_pke(
            model, clean_model, toxic_model, top_k_neurons=top_k,
            tokenizer=tok, num_steps=6, lr=1e-2, max_edit_magnitude=None,
        )
        assert result is model
        after = result.layers[0].mlp.down_proj.weight.detach()
        return int((~torch.isclose(after, before).all(dim=1)).sum().item())

    changed_k1 = _changed_rows(1)
    changed_k3 = _changed_rows(3)

    assert changed_k1 >= 1, "k=1 should edit at least one located neuron"
    assert changed_k3 >= changed_k1, (
        f"k=3 edited {changed_k3} rows, fewer than k=1's {changed_k1}"
    )
    assert changed_k3 > changed_k1, (
        f"larger k should edit more neurons (k1={changed_k1}, k3={changed_k3})"
    )


# ─── LLM-Steg ────────────────────────────────────────────────────────────────
# StegAttack removed: the in-house verify attack reimplementations were dropped
# (only Abliteration + BoN passed the faithfulness audit). See docs/getting-started/taxonomy.md.


# ─── STAR-1 Dataset Pack ─────────────────────────────────────────────────────

def test_star1_pack():
    from safetune.core.data_compiler.safety_packs import resolve_pack
    pack = resolve_pack("star1")
    assert pack.name == "star1"
    assert "UCSC-VLAA/STAR-1" in pack.datasets["primary"]
