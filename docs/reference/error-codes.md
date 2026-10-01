# Error Codes

Every structured SafeTune error is a `SafeTuneError` (`safetune.utils.errors`) carrying a stable `error_code`, a human `message`, and a list of `suggestions`. Catch the base class, or match on `error_code` to handle one kind specifically — never string-match the message text, which can change.

```python
from safetune.utils.errors import SafeTuneError, GatedResourceError

try:
    evaluate(model, benchmarks=["hexphi"])
except GatedResourceError as e:
    print(f"Need access to {e.repo_id} ({e.kind}): {e.suggestions[0]}")
except SafeTuneError as e:
    print(e.error_code, e.message)
```

| `error_code` | Exception class | Raised when |
|---|---|---|
| `CONFIG_ERROR` | `ConfigurationError` | An invalid config value — bad optimizer/scheduler name, unknown field. |
| `TRAINING_ERROR` | `TrainingError` | A training-time failure — OOM, NaN/Inf loss, a CUDA error. |
| `ENV_ERROR` | `EnvironmentError` | A missing or incompatible dependency (`torch`, `transformers`, `trl`, `bitsandbytes`, `unsloth`). |
| `VALIDATION_ERROR` | `ValidationError` | A field fails validation — wrong type, out of range, missing a required value. |
| `GATED_RESOURCE` | `GatedResourceError` | A Hugging Face model or dataset SafeTune itself needs (an eval judge, a benchmark dataset, a generation/steer backend's model) is gated or otherwise inaccessible with the current credentials. |

`GatedResourceError` additionally carries `repo_id` (the exact Hugging Face id) and `kind` (`"judge model"`, `"dataset"`, `"model"`, or `"auxiliary model"`) as real attributes, not just message text — read them to tell the caller exactly which access to request. It does not cover the model/adapter you pass in yourself to train or evaluate; only SafeTune's own internal dependencies raise it.
