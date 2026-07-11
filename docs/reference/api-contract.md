# SafeTune — API Interface Contract

## 1. Public Namespace

### 1.1 Top-Level (`safetune/__init__.py`)

```python
__all__ = [
    "recover", "unlearn", "harden", "steer",     # Tier 1 interventions
    "interpret", "evaluate",                      # Tier 2 instrumentation
    "core", "data", "rewards", "utils",           # supporting namespaces
]
```

```python
import safetune

# All six pillars accessible as attributes
safetune.harden.SafeGradTrainer(...)
safetune.recover.apply_resta(finetuned=..., base=..., aligned=...)
safetune.unlearn.rmu_unlearn(...)
safetune.steer.RefusalDirectionModel(model, direction=...)
safetune.interpret.safety_circuit_info(model, tokenizer, harmful, harmless)
safetune.evaluate.evaluate(model, benchmarks=["harmbench"])
```

### 1.2 Pillar Packages

#### `safetune.harden` — Train-time defenses

```python
__all__ = [
    "SafeGradTrainer", "SafeGradConfig",
    "LisaTrainer", "LisaConfig",
    "AsFTTrainer", "AsFTConfig", ...
]
class XxxTrainer(transformers.Trainer):
    """Input: model, train_dataset[, safety_dataset, reference_model]
       Output: trained model checkpoint"""
    def train(self, *args, **kwargs): ...
```

#### `safetune.recover` — Weight-space patching

```python
__all__ = [
    "apply_resta", "apply_lox", "apply_ctheta",
    "apply_safe_delta", "task_arithmetic", ...
]
def apply_resta(finetuned, base, aligned, alpha=1.0, ...) -> nn.Module:
    """Input: drifted (finetuned) model + base + aligned reference
       Output: patched model (same type as finetuned)"""
```

#### `safetune.unlearn` — Forget-set training

```python
__all__ = [
    "rmu_unlearn", "npo_unlearn", "flat_unlearn",
    "gradient_ascent_unlearn", "scrub_unlearn", ...
]
def rmu_unlearn(model, retain_batches, forget_batches,
                *, frozen_model=None, config=None) -> nn.Module:
```

#### `safetune.steer` — Inference-time steering

```python
__all__ = [
    "RefusalDirectionModel", "CAAModel", "CASTModel",
    "RefusalDirectionConfig", "CAAConfig", ...
    "extract_refusal_direction", "extract_caa_vectors", ...
]
```

#### `safetune.interpret` — Diagnostics

```python
__all__ = [
    "identify_safety_neurons", "SafetyNeuronConfig",
    "safety_circuit_info", "eap_safety_circuit", ...
]
```

#### `safetune.evaluate` — Measurement

```python
__all__ = [
    "evaluate", "AbliterationAttack", "BoNAttack",
    "TamperBenchEvaluator", "SpectralEntropyMonitor", ...
]
```

### 1.3 Runner API (`safetune.runner`)

The runner package wraps pillar methods in a uniform `Trainer` interface:

```python
trainer = harden.LisaTrainer(model, tokenizer, lisa_rho=0.1)
out_path = trainer.train(train_dataset, out_dir="./ckpt", safety_dataset=safety_ds)
metrics  = trainer.eval("run_name", out_path)
trainer.save_results(metrics, variant="default")
```

Contract:
| Method | Input | Output |
|---|---|---|
| `.train(data, out_dir)` | dataset, str | str (checkpoint path) |
| `.eval(folder, path)` | str, str | dict (metrics) |
| `.apply(**kwargs)` | method-specific overrides | nn.Module (patched model) |
| `.calibrate(harmful, harmless)` | list[str], list[str] | tuple(model, path) |
| `.unlearn(forget, retain)` | dataset, dataset | nn.Module |
| `.save_results(metrics)` | dict | None |
| `.evaluate(model, domain)` | nn.Module, str | dict |

#### `apply()` contract (recover trainers)

Every `_RecoverBase` subclass must implement `apply()` as follows:

- **Returns** the patched model (`nn.Module`). May mutate `self.model` in-place
  and return it, or return a new object.
- **Never writes to disk.** The caller is responsible for calling
  `save_checkpoint(patched, tokenizer, name)` separately.
- **Accepts method-specific keyword overrides** — e.g. `trainer.apply(strength=1.5)`.
- `self.model` is lazily loaded on first access; subclasses must not assume it
  is pre-loaded in `__init__`.

### 1.4 Algo Registry (`safetune.runner._registry`)

The CLI resolves `--algo` names to trainer class names via a central registry:

```python
from safetune.runner._registry import (
    HARDEN_REGISTRY, register_harden, register_recover, register_unlearn,
)

# Read-only inspection
print(HARDEN_REGISTRY["lisa"])   # → "LisaTrainer"

# Runtime extension (no library edits needed)
register_harden("mymethod", "MyTrainer")   # MyTrainer in safetune.runner.harden
register_recover("myrecover", "MyRecoverTrainer")
register_unlearn("myunlearn", "MyUnlearnTrainer")
```

The registry is the single source of truth for all alias → class name
mappings. Adding a new method to the library only requires:
1. Implement the trainer in the appropriate `runner/<pillar>/` module.
2. Re-export it from `runner/<pillar>/__init__.py`.
3. Add one line to `_registry.py`.

## 2. Input/Output Tensor Contract

All tensor operations follow PyTorch conventions:

| Parameter | Shape | Dtype | Notes |
|---|---|---|---|
| `input_ids` | `(B, seq_len)` | `torch.long` | Token IDs |
| `attention_mask` | `(B, seq_len)` | `torch.long` | 1 = attend, 0 = mask |
| `labels` | `(B, seq_len)` | `torch.long` | -100 = ignore |
| Hidden states | `(B, seq_len, D)` | `torch.float32/16` | D = model hidden dim |
| Direction vectors | `(D,)` | `torch.float32` | Unit refusal direction |
| CAA vectors | `(L, D)` | `torch.float32` | L layers × D dim |

## 3. Configuration Schema

### Method configs

Method configs are `dataclass`-based with defaults. Standalone configs (e.g.
`RMUConfig`, `LisaConfig`) hold the method-specific hyperparameters:

```python
@dataclass
class RMUConfig:
    layer_id: int = 7
    update_layer_ids: list | None = None
    steering_coeff: float = 20.0
    alpha: float = 100.0
    lr: float = 5e-5
    max_num_batches: int = 80
    param_substring: str = "mlp.down_proj"
```

Some HARDEN configs (e.g. `SafeGradConfig`) subclass
`transformers.TrainingArguments` and add no extra fields; the method-specific
knobs for those (e.g. SafeGrad's `rho`, `kl_temperature`) are keyword arguments
on the trainer's `__init__`, not config fields.

Config name pattern: `{MethodName}Config`.

### CLI / run config (`safetune.config.SafeTuneConfig`)

`SafeTuneConfig` is a flat dataclass that captures all CLI-level options and
loads from YAML:

```python
from safetune.config import SafeTuneConfig

cfg = SafeTuneConfig.from_yaml("run.yaml")   # unknown YAML keys → method_kwargs
kw  = cfg.as_trainer_kwargs()                # dict ready for **kwargs
ns  = cfg.to_namespace()                     # argparse.Namespace (all fields flat)
```

Key fields: `command`, `algo`, `model`, `base`, `aligned`, `output`, `epochs`,
`batch_size`, `lr`, `precision`, `optimizer`, `logging_steps`, `train_dataset`,
`train_split`, `dataset`, `drift_task`, `method_kwargs`.

Any YAML key not matching a standard field is collected into `method_kwargs` and
forwarded to the trainer constructor, so method-specific hyperparameters
(`lisa_rho`, `rank`, `inner_steps`, …) can be set from the same config file.

## 4. Backward Compatibility Policy

- SemVer since v1.0.0 (current: 1.0.0)
- `safetune.evaluate` is the Measure pillar; the old `verify` name was removed
  (not aliased — `import safetune.verify` raises `ModuleNotFoundError`)
- `safetune.core.unlearn` → `safetune.unlearn` (shim maintained at old path)
- `safetune.core.interpret` → `safetune.interpret` (shim maintained at old path)
- No breaking changes within the same major version.
- Deprecation of a public API element triggers a `DeprecationWarning` one minor
  version before removal.
