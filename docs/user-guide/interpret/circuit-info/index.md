# Circuit info

The `CircuitInfo` container holds safety-relevant units, layer suggestions,
and metadata. It is the output of both safety neuron identification and EAP
circuit discovery, and feeds Steer and localization-aware Recover methods.

| Component | Description |
|---|---|
| [safety_circuit_info()](safety-circuit-info.md) | Convenience wrapper — runs refusal direction + safety neurons |
| [CircuitInfo](circuit-info-object.md) | Universal circuit data container |
