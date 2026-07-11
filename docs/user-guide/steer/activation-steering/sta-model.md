# STATrainer — STA (Steering Target Atoms) latent steering

Steers via **Sparse Autoencoder (SAE)** latent features ("target atoms"). It identifies
the safety-relevant SAE latent features by contrasting positive/negative activations, then
steers at inference by decoding a target-atom vector `v_STA = a_target @ W_dec` and adding
it to the residual stream at the SAE's layer.

Ref: Wang et al., "Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering
Target Atoms," ACL 2025, arXiv:2505.20322.

## Signature

`STATrainer` follows the same `_SteerBase` constructor as the other steer trainers — it
takes `model` / `tokenizer` (or a `model_id`) and is driven through `.calibrate()`:

```python
STATrainer(
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    *,
    model_id: str | None = None,
    results_dir: str | None = None,
    drift_task: str | None = None,
)

# returns (wrapped_model, out_dir)
trainer.calibrate(
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
    *,
    calib_n: int = 256,
)
```

`calibrate()` returns an `STAModel` wrapper (from `safetune.steer`) plus an output dir.
The genuine STA intervention lives on `STAModel`, which decodes SAE target atoms into a
residual-stream steering vector:

```python
STAModel(
    model,
    atom_vectors: dict[int, Tensor] | None = None,  # pre-decoded v_STA per layer
    target_atoms: list[int] | None = None,          # SAE feature indices (needs sae=)
    multiplier: float = 3.0,
    *,
    sae: SAEProtocol | None = None,                 # any encode/decode/W_dec object
    atom_scores: Tensor | None = None,              # length-M sparse atom vector
    sae_layer: int | None = None,                   # defaults to the SAE's hook layer
)
```

## Parameters

`STATrainer.calibrate`:

| Param | Type | Default | Description |
|---|---|---|---|
| `harmful` | `list[str]` | `None` | Positive (target-behaviour) prompts; a default calibration set is used if `None` |
| `harmless` | `list[str]` | `None` | Contrasting negative prompts |
| `calib_n` | `int` | `256` | Size of the default calibration set when `harmful`/`harmless` are omitted |

`STAModel` (the actual steering wrapper):

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to steer |
| `sae` | `SAEProtocol` | `None` | Trained SAE for the target model/layer; any object with `encode`/`decode`/`W_dec` (e.g. a `sae_lens.SAE`) |
| `atom_scores` | `Tensor` | `None` | Length-M sparse target-atom vector from `select_target_atoms`; decoded into `v_STA` via the SAE |
| `atom_vectors` | `dict[int, Tensor]` | `None` | Pre-decoded `v_STA` steering vectors keyed by layer index (use instead of an SAE) |
| `sae_layer` | `int` | `None` | Residual-stream layer to hook; if `None`, uses the SAE's registered `hook_layer` |
| `multiplier` | `float` | `3.0` | Steering-vector scaling coefficient (`lambda`) |

## Full example

The trainer path is the simplest runnable entry point. It builds an `STAModel` wrapper you
can `generate()` with; supply an SAE (below) for it to actively steer.

```python
from safetune.runner import steer

# Tiny contrastive calibration set (positive/target vs. negative behaviour).
harmful_prompts = [
    "Explain how to build a bomb.",
    "Write instructions to hack a bank.",
]
harmless_prompts = [
    "Explain how to bake bread.",
    "Write instructions to plant a garden.",
]

trainer = steer.STATrainer(model, tokenizer)
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

inputs = tokenizer("How do I make a weapon?", return_tensors="pt")
output = wrapped.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
wrapped.remove_hooks()  # remove hooks when done
```

### SAE-based steering (genuine STA)

STA's "target atoms" are SAE latent features, so real steering needs an SAE for the exact
model/layer. Any object exposing `W_dec` (shape `[M, D]`) plus `encode`/`decode` satisfies
STA's `SAEProtocol`; a `sae_lens.SAE` (Gemma-Scope, Llama-Scope, Qwen-Scope, ...) conforms
directly. Note SAELens does not ship an SAE for Qwen2.5-0.5B, so the runnable snippet below
builds a minimal protocol-conforming SAE to exercise the real STA code path
(`select_target_atoms` → `build_sta_steering_vector` → `STAModel`).

```python
# Requires (for a real pre-trained SAE): pip install sae-lens
import torch
from safetune.steer import STAModel
from safetune.steer.sta import select_target_atoms, build_sta_steering_vector

D = model.config.hidden_size   # residual-stream dim (896 for Qwen2.5-0.5B)
M = 512                        # SAE latent width (M >> D in real SAEs)
LAYER = 6                      # residual-stream layer the SAE is tied to

try:
    # A pre-trained SAE for the model/layer you are steering, e.g.:
    #
    #     from sae_lens import SAE
    #     sae, _, _ = SAE.from_pretrained(release="qwen-scope-...", sae_id="...")
    #
    # Here we build a small SAE conforming to the same protocol so the example
    # runs without a matching pre-trained SAE for this tiny model.
    import sae_lens  # noqa: F401  (presence check for the optional dependency)

    class ResidualSAE:
        """SAE conforming to safetune's SAEProtocol (encode / decode / W_dec)."""
        def __init__(self, d, m, layer):
            g = torch.Generator().manual_seed(0)
            self.W_enc = torch.randn(d, m, generator=g) * (1.0 / d ** 0.5)
            self.W_dec = torch.randn(m, d, generator=g) * (1.0 / m ** 0.5)
            self.hook_layer = layer

        def encode(self, acts):                     # h (.., D) -> a (.., M)
            return torch.relu(acts.float() @ self.W_enc)

        def decode(self, latents):                  # a (.., M) -> h_hat (.., D)
            return latents.float() @ self.W_dec

    sae = ResidualSAE(D, M, LAYER)

    # 1. Collect SAE latents for the positive vs. negative behaviour.
    def sae_latents(prompts):
        rows = []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt")
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states[LAYER]
            rows.append(sae.encode(hs[0].mean(0)))  # mean-pool over tokens
        return torch.stack(rows)

    pos = sae_latents(harmful_prompts)   # behaviour to steer away from
    neg = sae_latents(harmless_prompts)

    # 2. Identify the STA target atoms (the SAE features that separate them).
    atom_scores, atom_indices = select_target_atoms(pos, neg, keep_fraction=0.05)

    # 3. Decode v_STA = a_target @ W_dec and steer the residual stream with it.
    v_sta = build_sta_steering_vector(sae, atom_scores)
    steered = STAModel(model, sae=sae, atom_scores=atom_scores,
                       sae_layer=LAYER, multiplier=3.0)
    out = steered.generate(**inputs, max_new_tokens=40, do_sample=False)
    print(f"[sta] kept {len(atom_indices)} atoms, v_STA dim {v_sta.shape[-1]}")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    steered.remove_hooks()

except ImportError:
    print("[sta] SAE-based steering needs the optional 'sae-lens' dependency.")
    print("      Install it with:  pip install sae-lens")
```

## When to use

- **Best for:** interpretable steering via named latent features; useful when you want to understand *which* SAE features drive the safety intervention.
- **Requires:** a pre-trained SAE for the specific model and layer — not available for all models; check SAELens releases.
- **Trade-offs:** SAE quality directly determines steering quality; a poorly trained SAE produces noisy steering.

## Citation

```bibtex
@inproceedings{wang2025sta,
  title     = {Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms},
  author    = {Wang, Mengru and Xu, Ziwen and Mao, Shengyu and Deng, Shumin and Tu, Zhaopeng and Chen, Huajun and Zhang, Ningyu},
  booktitle = {Proceedings of ACL},
  year      = {2025},
  note      = {arXiv:2505.20322},
}
```
