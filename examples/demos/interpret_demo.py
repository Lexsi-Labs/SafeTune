#!/usr/bin/env python3
"""SafeTune INTERPRET demo — locate safety-relevant neurons and circuits.

The Diagnose pillar (safetune.interpret) identifies *where* safety lives
inside a model's weights and activations. This demo covers the three main
entry points:

  • ``identify_safety_neurons`` — find MLP neurons that drive refusal
  • ``safety_circuit_info``   — build a CircuitInfo from contrast pairs
  • ``eap_safety_circuit``    — run EAP / EAP-IG for circuit discovery

Usage
-----
    python examples/demos/interpret_demo.py
"""
from __future__ import annotations

import sys


def main() -> int:
    print("SafeTune INTERPRET demo\n")

    # 1. Safety neurons — weight-based scoring via activation caching.
    print("[1/3] identify_safety_neurons")
    print("      Scans MLP neurons and returns those most correlated with refusal.")
    print("      import: from safetune.interpret import identify_safety_neurons, SafetyNeuronConfig\n")
    print("      cfg = SafetyNeuronConfig(top_k_per_layer=50)")
    print("      report = identify_safety_neurons(model, directions, cfg,")
    print("                                       tokenizer=tokenizer,")
    print("                                       harmful_prompts=harmful, harmless_prompts=harmless)")
    print("      # report is a SafetyNeuronReport (per-layer neuron indices + scores)\n")

    # 2. Circuit Info — unified container from contrastive analysis.
    print("[2/3] safety_circuit_info")
    print("      Builds a CircuitInfo object that can be passed to Steer or Recover.")
    print("      import: from safetune.interpret import safety_circuit_info\n")
    print("      circuit = safety_circuit_info(model, tokenizer, harmful, harmless)")
    print("      print(circuit.layer_suggestions)  # suggested layers to intervene on")
    print("      print(circuit.safety_units)    # identified safety-relevant units")
    print("      circuit.save('path/to/circuit.json')  # serialise for later use\n")

    # 3. EAP / EAP-IG — activation-based circuit discovery.
    print("[3/3] eap_safety_circuit")
    print("      Edge Attribution Patching locates the edges (MLP + attention)")
    print("      most responsible for the safety response.")
    print("      import: from safetune.interpret import eap_safety_circuit, EAPSafetyCircuitConfig\n")
    print("      cfg = EAPSafetyCircuitConfig(method='eap_ig', top_k_edges=100)")
    print("      circuit = eap_safety_circuit(model, tokenizer, harmful, harmless, cfg)")
    print("      # circuit is a CircuitInfo with edge-level attribution scores\n")

    # 4. CircuitInfo bridge to downstream pillars.
    print("Using CircuitInfo in downstream pillars:")
    print("  from safetune.core.neuron_safety import apply_circuit_to_safety_lora")
    print("  lora_cfg = apply_circuit_to_safety_lora(circuit)       # → Steer")
    print("  from safetune.core.neuron_safety import get_lora_targeting_from_circuit")
    print("  targeting = get_lora_targeting_from_circuit(circuit)   # → Recover\n")

    print("See docs/guides/interpret/ for the full catalog of methods.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
