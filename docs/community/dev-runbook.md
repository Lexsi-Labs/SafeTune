# SafeTune — Development & Enterprise Runbook

## 1. Local Developer Setup

### 1.1 One-liner

```bash
git clone https://github.com/Lexsi-Labs/SafeTune.git
cd SafeTune
pip install -e ".[dev,docs]"
```

### 1.2 Editable Install

The `pyproject.toml` uses `[tool.setuptools.packages.find] where = ["src"]`.
Install in editable mode so changes to `src/safetune/` take effect immediately:

```bash
pip install -e ".[dev]"       # + dev dependencies (pytest, ruff, mypy)
pip install -e ".[docs]"      # + mkdocs-material for docs
pip install -e ".[vllm]"      # + vLLM backend for eval
```

### 1.3 Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | auto | GPU selection |
| `HF_TOKEN` | — | HuggingFace authentication |
| `SAFETUNE_DRIFT_MODEL` | — | Path to drifted checkpoint |
| `SAFETUNE_BASE_MODEL` | — | Path to base model checkpoint |
| `SAFETUNE_ALIGNED_MODEL` | — | Path to aligned reference checkpoint |
| `SAFETUNE_GPU_MEM` | `0.75` | vLLM GPU memory utilization (also settable per-trainer via `gpu_memory_utilization=`) |

### 1.4 Quick Smoke Test

```bash
python -c "import safetune; print(safetune.__version__)"
python examples/quickstart/quickstart.py --device cpu  # requires ~1.5 GB RAM
```

## 2. Testing Matrix

### 2.1 Local (CPU)

```bash
pytest tests/ -m "not slow and not gpu" --timeout=120
```

Runs instantiation tests, data pipeline tests, and configuration validation.
GPU-dependent tests are skipped.

### 2.2 CI (GitHub Actions)

See `.github/workflows/smoke.yml`:
- `pip install -e ".[dev]"`
- `python -c "import safetune; print(safetune.__version__)"` (import check)
- `pytest tests/ -m "not gpu and not unsloth and not wandb" --no-cov -q --timeout=120`

### 2.3 Full Integration (GPU required)

The `tests/integration/` scripts run each pillar's trainers end-to-end
(`.train()` then `.evaluate()`) and print PASS / FAIL / SKIP. They are run
directly with `python` (not collected by `pytest`) and require external
checkpoints supplied via `SAFETUNE_DRIFT_MODEL` (and, for recover,
`SAFETUNE_ALIGNED_MODEL` / `SAFETUNE_BASE_MODEL`):

```bash
export SAFETUNE_DRIFT_MODEL=/path/to/drifted
export SAFETUNE_ALIGNED_MODEL=/path/to/aligned
python tests/integration/test_harden.py   # expects a CUDA GPU with 24+ GB VRAM
```

Trainers exercised per pillar:
- `test_harden.py` — 22 trainers
- `test_recover.py` — 24 trainers
- `test_steer.py` — 19 trainers
- `test_unlearn.py` — 6 trainers

Each train + eval cycle takes 5-30 minutes depending on method.

Cap the benchmark eval for a fast smoke run with `SAFETUNE_EVAL_LIMIT=N`
(N examples per lm-eval task).

### 2.3a Lightweight proxy mode (no checkpoints needed)

When you don't have the drifted checkpoints, run the pipeline end-to-end on a
small public model to prove every pillar's train/apply/calibrate/unlearn path
executes (this is a plumbing smoke, **not** a safety-drift measurement):

```bash
SAFETUNE_PROXY=1 CUDA_VISIBLE_DEVICES=1 pytest tests/integration -m proxy
```

Proxy mode defaults `SAFETUNE_DRIFT_MODEL` to `Qwen/Qwen2.5-0.5B-Instruct`
(+ small distinct base/aligned refs) and runs `tests/integration/test_proxy.py`
— one train/apply/calibrate/unlearn + generation per pillar, ~70s on one GPU.
`proxy`-marked tests only run under `SAFETUNE_PROXY=1`; the full-eval tests
(mode above) still require real checkpoints.

### 2.4 Testing Hardware Requirements

| Test level | GPU | VRAM | Time |
|---|---|---|---|
| CPU unit tests | None | 0 GB | 2 min |
| CI smoke tests | None | 0 GB | 5 min |
| Single-pillar integration | 1× A100 80GB | 24 GB | 30 min |
| Full suite | 1× A100 80GB | 80 GB | 2 hr |
| Evaluation benchmarks | 1× A100 80GB | 40 GB | 1 hr |

## 3. Code Quality Pipeline

### 3.1 Linting & Formatting

```bash
flake8 src/                           # style + lint (pinned in [dev])
ruff check src/                       # optional alternative linter (pip install ruff)
black --check src/                    # formatting
isort --check src/                    # import ordering
mypy src/ --ignore-missing-imports   # type checking (loose)
```

All configured in `pyproject.toml` under `[tool.ruff]`, `[tool.black]`,
`[tool.isort]`, `[tool.mypy]`. The `[dev]` extra pins flake8, black, isort, and
mypy; ruff has a `[tool.ruff]` config but is not installed by `[dev]`.

### 3.2 Pre-commit (optional)

```bash
pip install pre-commit
pre-commit install  # once per clone
```

Note: `.pre-commit-config.yaml` is `.gitignore`d — create your own from the
tool configs in `pyproject.toml`.

### 3.3 Documentation Build

```bash
pip install -e ".[docs]"
mkdocs build --strict      # builds site/ with zero warnings
mkdocs serve -a 0.0.0.0:8000  # live preview
```

## 4. CI/CD Pipeline

### 4.1 Workflow: Docs Build (`docs.yml`)

```yaml
on:
  push/PR → main:
    paths: ['docs/**', 'mkdocs.yml', 'requirements-docs.txt']
jobs:
  build:
    steps:
      - pip install -r requirements-docs.txt
      - mkdocs build --strict
```

### 4.2 Workflow: Smoke Tests (`smoke.yml`)

```yaml
on: push/PR → main
jobs:
  test:
    timeout-minutes: 25
    steps:
      - pip install -e ".[dev]"
      - python -c "import safetune; print(safetune.__version__)"
      - pytest tests/ -m "not gpu and not unsloth and not wandb" --no-cov -q --timeout=120
```

### 4.3 Versioning

- Current: `1.0.0` (SemVer)
- PyPI release: `python -m build && twine upload dist/*`
- Version pinned in `src/safetune/__init__.py`: `__version__ = "1.0.0"`
- Mirrored in `pyproject.toml`: `version = "1.0.0"`

### 4.4 Wheel Build

```bash
pip install build
python -m build
# Produces dist/safetune-1.0.0-py3-none-any.whl + dist/safetune-1.0.0.tar.gz
```

## 5. Dependency Management

### 5.1 Core Dependencies (`pyproject.toml`)

```
torch >= 2.7.0
transformers >= 4.48.3, < 6
trl >= 0.12, < 2
datasets >= 2.14
peft
accelerate
```

### 5.2 Optional Extras

| Extra | Key deps | Purpose |
|---|---|---|
| `[dev]` | pytest, black, isort, flake8, mypy, pre-commit | Development tooling |
| `[docs]` | mkdocs, mkdocs-material, mkdocs-mike | Documentation build |
| `[vllm]` | vllm | vLLM evaluation backend |
| `[interpret]` | transformer_lens | Interpretability tools |
| `[viz]` | matplotlib, seaborn | Visualization |

### 5.3 Docker (Optional)

For reproducible GPU environments:

```dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
COPY . /app
WORKDIR /app
RUN pip install -e ".[dev,vllm]"
CMD ["pytest", "tests/"]
```

## 6. Adding a New Method

Adding a method requires touching exactly three places.

### 6.1 Implement the trainer

Create (or extend) a module in the appropriate `runner/<pillar>/` directory.
Inherit from the pillar base class and implement the required method:

```python
# src/safetune/runner/harden/_mymethod.py
from ._base import _HardenBase, _keep_model_columns, default_data_collator

class MyMethodTrainer(_HardenBase):
    """One-line description.

    Args:
        alpha: my hyperparameter. Default 1.0.
    """
    METHOD = "MyMethod"

    def __init__(self, model=None, tokenizer=None, *, alpha: float = 1.0, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.alpha = alpha

    def train(self, train_dataset, out_dir: str = None, **kwargs) -> str:
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        # ... training logic ...
        return self._save_merged(model, out_dir)
```

For **recover** trainers, inherit from `_RecoverBase` and implement `apply()`
following the contract in [api-contract.md](../reference/api-contract.md):

```python
# src/safetune/runner/recover/_myrecover.py
from ._base import _RecoverBase
import safetune.recover as R

class MyRecoverTrainer(_RecoverBase):
    METHOD = "MyRecover"

    def apply(self, **kwargs):
        return R.my_recover_fn(self.model)
```

### 6.2 Re-export from `__init__.py`

Add the class to the pillar runner's `__init__.py` so it is importable via
`from safetune.runner import harden`:

```python
# src/safetune/runner/harden/__init__.py
from ._mymethod import MyMethodTrainer

__all__ = [..., "MyMethodTrainer"]
```

### 6.3 Register in `_registry.py`

Add one line to `src/safetune/runner/_registry.py`:

```python
HARDEN_REGISTRY: dict[str, str] = {
    ...
    "mymethod": "MyMethodTrainer",   # ← add this
}
```

After this, `safetune train --algo mymethod` and `safetune list` both work.
No other files need to change.

### 6.4 Runtime registration (third-party extensions)

External code can extend the registry without editing library files:

```python
from safetune.runner._registry import register_harden
register_harden("mymethod", "MyMethodTrainer")
# Then run the CLI or use the runner as normal
```

### 6.5 Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | auto | GPU selection |
| `HF_TOKEN` | — | HuggingFace authentication |
| `SAFETUNE_DRIFT_MODEL` | — | Path to drifted checkpoint |
| `SAFETUNE_BASE_MODEL` | — | Path to base model checkpoint |
| `SAFETUNE_ALIGNED_MODEL` | — | Path to aligned reference checkpoint |
| `SAFETUNE_GPU_MEM` | `0.75` | vLLM GPU memory utilization (also settable per-trainer via `gpu_memory_utilization=`) |

## 7. Disaster Recovery & Fail-Safe

| Scenario | Mitigation |
|---|---|
| GPU OOM during training | Reduce `per_device_train_batch_size`, enable gradient checkpointing |
| GPU OOM during vLLM eval | Reduce `gpu_memory_utilization` (env var `SAFETUNE_GPU_MEM`), default 0.75 |
| vLLM unsupported model | Auto-fallback to HuggingFace generation with warning |
| Transformers TF probe error | SafeTune disables TF/Flax on import (`USE_TF=0`) |
| Model download failure | Check HF token, network, disk space in `~/.cache/huggingface/` |
| Corrupted checkpoint | Delete checkpoint and re-run training |
