# SafeTune — Feature Map

Faithfulness-audit map for SafeTune's methods. The **Faithful?** column shows
the verdict for each audited component.

For an explanation of each verdict, see [Scope & Limitations](../community/scope.md).

## Audit status (v1.0.0) { #audit-status-v100 }

A full pre-release code audit (2026-07) found and **fixed** a set of defects
where several methods silently did less than documented: wrong-quantity math,
gradient-accumulation breakage, and wiring/default no-ops. All fixes ship in
v1.0.0.

Two changes affect how you call the fixed methods:

- **Some methods now require an input they previously ignored.** SafeDecoding
  and Contrastive Decoding raise on the HF path unless you pass a distinct
  expert / weak model (passing the target as its own guide was an identity
  no-op); AsFT and LoX-Harden warn when `aligned_model_path` is unset (the
  aligned reference is mandatory — without it they reduce to plain SFT);
  Surgery warns and is inactive unless the model is loaded with
  `attn_implementation="eager"`; SPPFT `sppft_mode="scale"` now falls back to
  the paper-faithful `"freeze"`.
- **The training-loop fixes were GPU-validated for execution.** Every fixed
  method runs one real `train()`/`apply()` step on GPU (Qwen2.5-0.5B) without
  crash and with its mechanism active, across the Python / CLI / YAML surfaces.
  What this does *not* yet cover is reproducing the safety-drift / utility
  numbers behind the "Faithful" badges — those need the real drifted 8B
  checkpoints (`SAFETUNE_DRIFT_MODEL`) run through `tests/integration`.

---

## Recover (26 entry points — weight-space recovery)

| Method | Faithful? |
|---|---|
| apply_ctheta + _from_state_dicts + sweep_ctheta_strength | Faithful |
| apply_prepost_merge | Faithful |
| task_arithmetic | Faithful |
| somf_merge | Variant |
| apply_safemerge | Faithful |
| apply_resta | Faithful |
| apply_lox | Faithful |
| apply_safe_lora | Faithful |
| apply_safe_delta | Simplified |
| apply_antidote | Faithful |
| apply_mscp | Faithful |
| apply_nlsr | Faithful |
| apply_pke | Faithful |
| apply_safereact | Faithful |
| apply_qresafe | Faithful |
| apply_aaq | Faithful |
| apply_lssf | Faithful |
| apply_deeprefusal | Variant |
| scrub_unlearn | Faithful |
| tracin_influence | Faithful |
| apply_wise_ft | Faithful |
| apply_safety_vector_restore | Faithful |
| apply_grad_selective_recover | Faithful |
| apply_oneshot_safety_patch | Faithful |
| apply_antidote_v2 | Faithful |
| apply_repnoise_recover | Faithful |

## Unlearn (6 entry points)

| Method | Faithful? |
|---|---|
| RMU (Representation Misdirection for Unlearning) | Faithful |
| NPO (Negative Preference Optimization) | Faithful |
| Gradient-Ascent / GradDiff | Faithful |
| crisp_unlearn | Variant |
| flat_unlearn | Faithful |
| simdpo_unlearn | Faithful |

## Harden (27 entry points)

| Method | Faithful? |
|---|---|
| CSTTrainer | Faithful |
| EMACallback | Faithful |
| SafeGradTrainer | Faithful |
| SPPFTTrainer | Faithful |
| DeRTaTrainer | Faithful |
| ASRTCallback | Variant |
| MARTTrainer | Faithful |
| DeepRefusalTrainer | Faithful |
| LisaTrainer | Faithful |
| AsFTTrainer | Faithful |
| STARDSSTrainer | Faithful |
| SAPTrainer | Faithful |
| DOORTrainer | Faithful |
| LookAheadTrainer | Faithful |
| AntibodyTrainer | Faithful |
| vaccine_loss | Faithful |
| SurgeryTrainer | Faithful |
| booster_project | Faithful |
| SaLoRA | Faithful |
| tar_outer_loss | Faithful |
| RepNoiseTrainer | Faithful |
| SEAMTrainer | Faithful |
| CTRAPTrainer | Faithful |
| TVaccineTrainer | Faithful |
| SEALTrainer | Faithful |
| ConstrainedSFTTrainer | Faithful |
| LoXHardenTrainer | Faithful |

## Steer (19 entry points)

| Method | Faithful? |
|---|---|
| extract_refusal_direction + RefusalDirectionModel + orthogonalize/restore | Faithful |
| CAA (extract_caa_vectors + CAAModel) | Faithful |
| LinearProbeGuardModel | Faithful |
| ContrastiveDecodingProcessor | Faithful |
| ProxyTuningProcessor | Faithful |
| SafeDecodingProcessor | Faithful |
| AlphaSteerModel | Faithful |
| RepBendModel | Faithful |
| TARModel | Faithful |
| SafeSteerModel | Faithful |
| SafeSwitchModel | Faithful |
| SCANSModel | Faithful |
| STAModel | Faithful |
| CircuitBreakerModel | Faithful |
| CircuitBreakerRRModel | Faithful |
| NudgingProcessor | Faithful |
| AdaSteerModel | Faithful |
| RRFAEnsemble | Faithful |
| CASTModel + fit_cast_probe | Faithful |

## Evaluate — Redteam

| Method | Faithful? |
|---|---|
| AbliterationAttack | Faithful |
| BoNAttack | Faithful |

## Evaluate — Infra (judges, loaders, benchmarks)

| Feature | Faithful? |
|---|---|
| StringMatchJudge | Faithful |
| HFJudge (HarmBench classifier) | Faithful |
| OpenAIJudge (StrongREJECT) | Faithful |
| JudgeAdapter | Faithful |
| SpectralEntropyMonitor | Faithful |
| evaluate.suite.evaluate | Faithful |
| TamperBenchEvaluator | Faithful |
| Loader: advbench | Faithful |
| Loader: xstest | Faithful |
| Loader: sorrybench | Faithful |
| Loader: orbench | Faithful |
| Loader: harmbench | Faithful |
| Loader: agentharm | Faithful |
| Loader: saladbench | Faithful |
| Loader: airbench | Faithful |
| Loader: cares | Faithful |
| Loader: wildjailbreak | Faithful |
| Loader: jailbreakbench | Faithful |
| Loader: muse | Faithful |
| Loader: rwku | Faithful |
| Loader: safedialbench | Faithful |
| Pareto utilities | Faithful |

## Interpret (6 tools)

| Feature | Faithful? |
|---|---|
| identify_safety_neurons (weight mode) | Faithful |
| identify_safety_neurons (activation mode) | Variant |
| safety_circuit_info | Faithful |
| CircuitInfo (+ JSON/YAML reader+writer) | Faithful |
| CircuitInfoProvider + save_circuit_info_to_file | Faithful |
| eap_safety_circuit (EAP / EAP-IG) | Faithful |

## Rewards

| Feature | Faithful? |
|---|---|
| SafetyRewardFunction | Faithful |
| SafetyReward | Simplified |
| ToxicityReward | Simplified |
| BiasReward / HarmlessnessReward / HallucinationReward | Variant |
| text / nlp / math / code / specialized rewards | mixed |

## Runtime guards

| Feature | Faithful? |
|---|---|
| redact_text / SafetyAuditEvent | Faithful |
| TenantPolicyRouter | Faithful |
| SafetyMiddleware | Faithful |
| InputSanitizer | Faithful |
| OutputVerifier | Faithful |
| AuditLogger / GuardrailPipeline | Faithful |
| CoSAlignFormatter | Faithful |
| CoSARuntime | Faithful |
| LLMSafeguardPredictor | Faithful |
| HookedGenerationWrapper / DynamicPatchingConfig | Faithful |

---

Variant methods run correctly but must not be cited as their named paper
method — see [Scope & Limitations](../community/scope.md) for details on each one.
