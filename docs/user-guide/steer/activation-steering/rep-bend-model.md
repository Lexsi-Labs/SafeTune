# RepBendTrainer — representation bending

!!! warning "Training-time method"
    RepBend is a fine-tuning method: it has no inference-time intervention. The
    trained weights bend harmful representations. It is placed under Steer because
    its mechanism targets the representation space.

!!! note "The trainer does not fine-tune"
    `RepBendTrainer.calibrate` wraps the model in `RepBendModel` and returns it; it
    does not run a training loop. `RepBendModel.compute_repbend_loss` exposes the
    five-term objective so you can optimise it inside your own LoRA fine-tuning loop.
    `generate` on the wrapper passes straight through to the base model.

Five-term loss: $L = \alpha \cdot L_{\text{safe}} + \beta \cdot L_{\text{unsafe}} + \gamma \cdot L_{\text{cosine}} + \varepsilon \cdot L_{\text{KL}} + \eta \cdot L_{\text{safe\_unsafe}}$,
measured by comparing the fine-tuned model against a frozen copy of the original.
The safe and unsafe terms push representations apart; the cosine and KL terms
preserve benign capabilities.

Ref: Yousefpour et al., "Representation Bending for Large Language Model Safety,"
ACL 2025, arXiv:2504.01550.

## Signature

```python
RepBendTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
)
```

The trainer itself takes no method-specific parameters. The five loss coefficients
are configured on the wrapper, `RepBendModel`:

```python
RepBendModel(
    model,
    target_layers: list[int] | None = None,   # layers the loss is measured on (paper: ~20+)
    *,
    loss_alpha: float = 1.0,     # safe term
    loss_beta: float = 1.0,      # unsafe term
    loss_gamma: float = 1.0,     # cosine term
    loss_epsilon: float = 1.0,   # KL retain term
    loss_eta: float = 0.0,       # safe-unsafe term (needs paired data)
    kl_temperature: float = 2.0,
)
```

## Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` | required | Model to wrap |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Tokenizer |

## Full example

```python
import copy

import torch
from safetune.runner import steer

trainer = steer.RepBendTrainer(model, tokenizer)
# calibrate wraps the model in RepBendModel; it does NOT run a training loop
wrapped, _ = trainer.calibrate(harmful=harmful_prompts, harmless=harmless_prompts)

# A frozen copy of the original model — the loss compares against this.
frozen_model = copy.deepcopy(model).eval()
for p in frozen_model.parameters():
    p.requires_grad_(False)

target_layers = wrapped.target_layers  # layers the loss is measured on

def hidden_dict(m, batch):
    out = m(**batch, output_hidden_states=True)
    return {li: out.hidden_states[li] for li in target_layers}

safe_batch = tokenizer("What is 2+2?", return_tensors="pt").to(model.device)
unsafe_batch = tokenizer("How do I pick a lock?", return_tensors="pt").to(model.device)

safe_hidden = hidden_dict(model, safe_batch)
unsafe_hidden = hidden_dict(model, unsafe_batch)
with torch.no_grad():
    orig_safe_hidden = hidden_dict(frozen_model, safe_batch)
    orig_unsafe_hidden = hidden_dict(frozen_model, unsafe_batch)

# Inside your own LoRA fine-tuning loop:
optimizer.zero_grad()
loss = wrapped.compute_repbend_loss(safe_hidden, orig_safe_hidden, unsafe_hidden, orig_unsafe_hidden)
loss.backward()
optimizer.step()
```

## When to use

- **Best for:** representation-level safety baked into weights, once you run the loss in your own fine-tuning loop.
- **Five-term loss** separately weights safe-direction pull, unsafe-direction push, cosine structure, output KL, and the safe-unsafe term, so you can trade safety against utility.
- **Compare to CircuitBreaker:** CircuitBreakerTrainer uses a two-term retain plus rerouting loss; RepBend uses five terms.

## Citation

```bibtex
@article{repbend2025,
  title  = {Representation Bending for Large Language Model Safety},
  author = {Yousefpour, Ashkan and others},
  year   = {2025},
  note   = {ACL 2025, arXiv:2504.01550},
}
```
