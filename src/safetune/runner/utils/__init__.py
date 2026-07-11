"""Runner utilities — ported from the SafeTune audit harness.

Re-exports the most-used helpers so callers can do:
    from safetune.runner.utils import eval_safety, load_model, build_sft_dataset
"""

from safetune.runner.utils.model_utils import (
    load_model,
    load_model_cpu,
    load_tok,
    free,
    lora_wrap,
    save_checkpoint,
)
from safetune.runner.utils.data_utils import (
    build_sft_dataset,
    build_safety_dataset,
    unlearn_forget_retain,
    refusal_prompt_pairs,
    refusal_prompt_pairs_large,
    harden_contamination_sets,
    harden_contamination_pairs,
    safety_raw_examples,
    # recover calibration
    recover_calib_input_ids,
    make_batch,
    # unlearn
    make_simdpo_pairs,
    # harden collators / tokenizers
    stardss_collator,
    derta_collator,
    derta_tokenize,
    sap_contrastive_dataset,
    # steer bench loading
    load_bench_prompts,
    SAFETY_BENCHES,
    UTILITY_TASKS_CORE,
    UTILITY_TASKS_BY_DRIFT,
)
from safetune.runner.utils.eval_runner import (
    eval_safety,
    eval_utility,
    safety_metrics,
    utility_metrics,
    all_metrics,
    safety_mean,
    utility_mean,
)
from safetune.runner.utils.results_writer import (
    ResultsWriter,
    DEFAULT_RESULTS_DIR,
)

__all__ = [
    # model
    "load_model", "load_model_cpu", "load_tok", "free", "lora_wrap",
    "save_checkpoint",
    # data — shared / smoke sets
    "build_sft_dataset", "build_safety_dataset",
    "refusal_prompt_pairs", "refusal_prompt_pairs_large",
    "safety_raw_examples",
    "SAFETY_BENCHES", "UTILITY_TASKS_CORE", "UTILITY_TASKS_BY_DRIFT",
    # data — harden
    "harden_contamination_sets", "harden_contamination_pairs",
    "stardss_collator", "derta_collator", "derta_tokenize",
    "sap_contrastive_dataset",
    # data — recover
    "recover_calib_input_ids", "make_batch",
    # data — unlearn
    "unlearn_forget_retain", "make_simdpo_pairs",
    # data — steer
    "load_bench_prompts",
    # eval
    "eval_safety", "eval_utility", "safety_metrics", "utility_metrics",
    "all_metrics", "safety_mean", "utility_mean",
    # results
    "ResultsWriter", "DEFAULT_RESULTS_DIR",
]
