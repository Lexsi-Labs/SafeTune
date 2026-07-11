# Configuration

`SafeTuneConfig` is the shared configuration object behind the CLI's `--config`
flag and the YAML workflow. Load it from YAML or build it in Python; every field
maps to a CLI flag, and explicit CLI flags override config values.

```python
from safetune.config import SafeTuneConfig

cfg = SafeTuneConfig.from_yaml("run.yaml")
```

See the [CLI Reference](../cli.md) for the YAML schema and precedence rules.

## Reference

::: safetune.config.SafeTuneConfig
    options:
      show_source: false
      heading_level: 3
      members_order: source
