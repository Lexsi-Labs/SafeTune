# Spectral Entropy Monitor

Detect steganographic / hidden-state-encoded jailbreaks by watching the
information content of the residual stream via spectral (von Neumann) entropy.

```python
from safetune.evaluate import SpectralEntropyMonitor

mon = SpectralEntropyMonitor(model, tokenizer)
baseline = mon.calibrate(benign_prompts=["What is 2+2?", "Tell me a joke."])
flags = mon.scan(["How to make a bomb?"])
for prompt_idx, layer_idx, entropy, z_score in flags:
    print(f"prompt {prompt_idx}, layer {layer_idx}: entropy={entropy:.3f} z={z_score:.2f}")
```

## SpectralMonitorConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `target_layers` | `Optional[List[int]]` | `None` | Layers to monitor |
| `z_threshold` | `float` | `2.0` | Flag when z-score < `-z_threshold` |
| `eigenvalue_floor` | `float` | `1e-12` | Numerical floor |
| `batch_size` | `int` | `8` | Forward-pass batch size |
| `skip_first_token` | `Optional[bool]` | `None` | Leave each prompt's first token (the attention sink) out of the SVD. On Qwen2.5-0.5B its hidden state has about 100x the norm of the other tokens, which made the entropy about 0.01 for every prompt |
| `chat_template` | `Optional[bool]` | `None` | Format each prompt as a chat user turn, as generation sees it |

`None` for `skip_first_token` or `chat_template` means `True`, or `False` under
`safetune.configure(legacy_spectral_monitor=True)` (the old monitor: first token
kept, raw prompts).

## API

| Method | Returns | Description |
|---|---|---|
| `.calibrate(benign_prompts)` | `Dict[int, Tuple[float, float]]` | Per-layer (mean, std) of entropy |
| `.scan(prompts)` | `List[Tuple[int, int, float, float]]` | `(prompt_idx, layer_idx, entropy, z_score)` |
| `.entropy_trajectory(prompt)` | `Dict[int, float]` | One prompt, no thresholding |

## Monitoring a fine-tune for safety drift

Calibrate on the same probe prompts you will scan, with the model before the
fine-tune, then scan each checkpoint with those prompts:

```python
mon = SpectralEntropyMonitor(aligned_model, tokenizer)
mon.calibrate(probe_prompts)          # baseline: these prompts, model before the fine-tune
for ckpt in checkpoints:
    mon.model = ckpt
    flags = mon.scan(probe_prompts)
```

A flag means the hidden states moved; any fine-tune can cause that, so check
refusal when a checkpoint is flagged.

## When to use

Use as a defensive primitive alongside text-level graders (e.g. StrongREJECT).
Not every jailbreak shows a spectral signature. Treat flagged pairs as
candidates for human review, not binary verdicts.
