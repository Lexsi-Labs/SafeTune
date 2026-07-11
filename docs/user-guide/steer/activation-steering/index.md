# Activation-steering methods

Add or remove safety-related directions in the residual stream. Most methods wrap a
frozen model with forward hooks that modify hidden states at inference time; a few
(CircuitBreaker, CircuitBreakerRR) are training-time methods grouped here because they
also target hidden representations.

| Method | Mechanism | Reference |
|---|---|---|
| [RefusalDirectionTrainer](refusal-direction-model.md) | Single direction steer/ablate | Arditi et al., NeurIPS 2024 |
| [CAATrainer](caa-model.md) | Contrastive activation addition | Panickssery et al., 2023 |
| [AdaSteerTrainer](ada-steer-model.md) | Adaptive two-direction steering | Zhao et al., EMNLP 2025 |
| [SafeSteerTrainer](safe-steer-model.md) | Category-routed steering | 2025 |
| [AlphaSteerTrainer](alpha-steer-model.md) | Closed-form ridge regression | 2025 |
| [SafeSwitchTrainer](safe-switch-model.md) | Two-stage prober + refusal head | Han et al., EMNLP 2025 |
| [SCANSTrainer](scans-model.md) | Adaptive-sign refusal steering | Cao et al., AAAI 2025 |
| [STATrainer](sta-model.md) | SAE latent steering | Wang et al., ACL 2025 |
| [CircuitBreakerTrainer](circuit-breaker-model.md) | Representation rerouting | Zou et al., NeurIPS 2024 |
| [CircuitBreakerRRTrainer](circuit-breaker-rr-model.md) | Dual-purpose RR | Zou et al., NeurIPS 2024 |
| [RepBendTrainer](rep-bend-model.md) | Representation bending | Yousefpour et al., ACL 2025 |
| [LinearProbeGuardTrainer](linear-probe-guard-model.md) | Probe-based guarding | |
| [CASTTrainer](cast-model.md) | Conditional activation steering | Wu et al., ICLR 2025 |
| [TARSteerTrainer](tar-model.md) | Tamper-resistant safeguards | Tamir et al., 2024 |
| [RRFAEnsembleTrainer](rrfa-ensemble.md) | RR for agentic safety | Zhou et al., 2025 |
