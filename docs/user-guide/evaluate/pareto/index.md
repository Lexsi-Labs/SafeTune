# Pareto Frontier Visualizer

Collect `(safety, capability)` pairs, identify the Pareto frontier, and plot
the trade-off.

```python
from safetune.evaluate.suite.pareto import ParetoVisualizer

viz = ParetoVisualizer(safety_label="refusal_rate", capability_label="gsm8k_acc")
viz.add("baseline", safety=0.85, capability=0.30)
viz.add("steer", safety=0.92, capability=0.27)

print(viz.summary())      # table with "on_frontier" flag
viz.plot("pareto.png")    # matplotlib scatter
```

## ParetoPoint

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Method identifier |
| `safety` | `float` | Safety score |
| `capability` | `float` | Capability score |
| `metadata` | `Dict` | Extra metadata |

## ParetoVisualizer API

| Method | Returns | Description |
|---|---|---|
| `.add(label, safety, capability, **metadata)` | `None` | Add one point |
| `.extend(items)` | `None` | Add tuples |
| `.frontier()` | `List[ParetoPoint]` | Non-dominated points |
| `.summary()` | `List[Dict]` | Table with `on_frontier` flag |
| `.to_json(path)` | `str` | Serialize to JSON |
| `.plot(path, title, annotate)` | `None` | Scatter plot (matplotlib) |

## When to use

Whenever you compare multiple defense methods or configs. A single Pareto
figure communicates the safety-utility trade-off more honestly than a bar
chart.
