"""Harden runner — CST, MART, DeepRefusal and Antibody.

These four shipped as "Python API only": their interfaces — TRL DPO pairs, a
co-evolution loop over two models, a refusal direction, SAM-style
harmful/refusal iterators — don't fit the uniform CLI train contract, so
`safetune train --algo cst` couldn't reach them (issue #21).

Each adapter here derives what it needs from the same BeaverTails harden
pairs every other runner trainer uses (``harden_contamination_pairs`` for raw
QA pairs, ``harden_contamination_sets`` for tokenized rows,
``refusal_prompt_pairs_large`` for the harmful/harmless prompt contrast), so
the CLI path works out of the box. Every method-specific piece stays a kwarg,
so Python callers keep full control and the documented intervention classes
stay reachable through their own modules (``safetune.harden.cst.CSTTrainer``,
``safetune.harden.mart.MARTTrainer`` …).
"""
import copy
import logging

from safetune.config import get_config, resolve_device
import safetune.harden as HARD
# The intervention classes come from their own modules, not from `HARD.<Name>`:
# the package attribute of those four names now resolves to *these* adapters,
# and the configs can be None when an optional dependency is missing.
from safetune.interventions.harden.cst import CSTTrainer as _CSTHF
from safetune.interventions.harden.deeprefusal import DeepRefusalTrainer as _DeepRefusalHF
from safetune.interventions.harden.antibody import AntibodyTrainer as _AntibodyHF

from ._base import _HardenBase, _precision
from safetune.runner.utils.model_utils import free

logger = logging.getLogger(__name__)


# ── Shared data helpers ──────────────────────────────────────────────────────

def _raw_pairs():
    """(contamination, refusal) raw QA pairs.

    ``refusal[i]`` is the safe response to ``contamination[i]``'s prompt
    (see ``_harden_raw_pairs``), so the two zip into preference triples.
    """
    from safetune.runner.utils.data_utils import harden_contamination_pairs
    contamination, _, refusal = harden_contamination_pairs()
    return contamination, refusal


def _tokenized_splits(tok):
    """Tokenized (contamination, refusal_aux) HF Datasets (input_ids /
    attention_mask / labels)."""
    from safetune.runner.utils.data_utils import harden_contamination_sets
    contamination, _tar_adversary, refusal_aux = harden_contamination_sets(tok)
    return contamination, refusal_aux


def _as_torch(dataset):
    """Torch-formatted view of a tokenized HF Dataset.

    ``harden_contamination_sets`` returns plain rows; the DeepRefusal /
    Antibody iterators index tensors (``model(**batch).loss``), so an
    unformatted dataset raises inside the trainer and the term is skipped —
    silently training benign-only. Same ``.with_format("torch")`` the
    intervention docstrings show.
    """
    from datasets import Dataset
    if isinstance(dataset, Dataset) and dataset.format.get("type") != "torch":
        return dataset.with_format("torch")
    return dataset


def _refusal_reward(prompt, response):
    """Default MART safety reward: 1.0 when the response opens with one of the
    AdvBench refusal prefixes, else 0.0."""
    text = (response or "").strip()
    prefixes = get_config().advbench_refusal_prefixes
    return 1.0 if any(text.startswith(p) for p in prefixes) else 0.0


def _report_to(wandb):
    return "wandb" if wandb else "none"


def _training_kwargs(self, out_dir, **extra):
    """TrainingArguments kwargs every adapter shares (precision from the
    runtime dtype, seed from the runtime config)."""
    kw = dict(
        output_dir=out_dir,
        num_train_epochs=self.epochs,
        per_device_train_batch_size=self.batch_size,
        learning_rate=self.lr,
        logging_steps=self.logging_steps,
        report_to=_report_to(self.wandb),
        **_precision(self.bf16, self.fp16),
    )
    seed = get_config().seed
    if seed is not None:
        kw["seed"] = seed
    kw.update(extra)
    return kw


# ── CST: contrastive safety training (TRL DPO) ───────────────────────────────

class CSTTrainer(_HardenBase):
    """CST: DPO over two opposite system prompts (safe / uncensored), so one
    checkpoint serves both behaviours.

    CST is data-centric — it needs preference rows, not an SFT task set. When
    the CLI passes its tokenized task set (or nothing), the adapter pairs the
    harden contamination set (unsafe_response) with the matched refusal set
    (safe_response) and formats them with ``prepare_cst_dataset``. Pass
    ``cst_examples=[{"prompt", "safe_response", "unsafe_response"}, …]`` to
    control the data.

    Note: without a PEFT adapter TRL deep-copies the model as the DPO
    reference, so peak memory is ~2x the model (the C7 constraint in #21).

    Args:
        beta: DPO temperature. Default 0.1.
        max_len: sequence budget for the pairs. Defaults to the runtime max_len.
        include_uncensored_pairs: also train the uncensored-side pairs.
            Default True (both system prompts are what makes CST controllable).
        safe_system_prompt / uncensored_system_prompt: override CST's system
            prompts.
    """
    METHOD = "CST"
    HF_TRAINER = _CSTHF

    def __init__(self, model=None, tokenizer=None, *,
                 beta: float = 0.1,
                 max_len: int = None,
                 include_uncensored_pairs: bool = True,
                 safe_system_prompt: str = None,
                 uncensored_system_prompt: str = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.beta = beta
        self.max_len = max_len or get_config().max_len or 1024
        self.include_uncensored_pairs = include_uncensored_pairs
        self.safe_system_prompt = safe_system_prompt
        self.uncensored_system_prompt = uncensored_system_prompt

    def train(self, train_dataset=None, out_dir: str = None, *,
              cst_examples=None, **kwargs) -> str:
        from datasets import Dataset
        from safetune.harden.cst import prepare_cst_dataset

        if cst_examples is None:
            if train_dataset is not None:
                logger.warning(
                    "CST: train_dataset carries no preference rows; using the "
                    "harden contamination/refusal pairs. Pass cst_examples= to "
                    "control the data."
                )
            contamination, refusal = _raw_pairs()
            cst_examples = [
                {"prompt": prompt,
                 "safe_response": refusal[i][1],
                 "unsafe_response": answer}
                for i, (prompt, answer) in enumerate(contamination)
            ]
        pairs = prepare_cst_dataset(
            cst_examples,
            safe_system_prompt=self.safe_system_prompt,
            uncensored_system_prompt=self.uncensored_system_prompt,
            include_uncensored_pairs=self.include_uncensored_pairs,
        )
        out_dir = self._resolve_out_dir(out_dir)
        # TRL's DPOConfig defaults max_prompt_length to 512 and rejects
        # max_length < max_prompt_length, so clamp to the default prompt budget.
        max_length = max(self.max_len, 512)
        cfg = HARD.CSTConfig(**_training_kwargs(self, out_dir, beta=self.beta,
                                                max_length=max_length))
        trainer = self.HF_TRAINER(
            model=self.model,
            args=cfg,
            train_dataset=Dataset.from_list(pairs),
            processing_class=self.tok,
        )
        trainer.train()
        return self._save_merged(self.model, out_dir)


# ── MART: multi-round adversarial red-teaming ───────────────────────────────

class MARTTrainer(_HardenBase):
    """MART: co-evolves an adversarial model against the target, keeping only
    target responses that reach ``safety_threshold`` (Zou et al.).

    The method trains two models, so the CLI contract (one model in) can't
    supply the attacker: the adapter deep-copies the target as ``M_adv``
    (the documented usage) unless ``adv_model=`` is passed. Seed prompts are
    the harmful contamination prompts; ``safety_reward_fn`` defaults to a
    refusal-prefix scorer over the configured AdvBench prefixes.

    ``train_dataset`` is ignored (like DOOR's) — MART is driven by
    ``seed_prompts`` and the reward, not by an SFT task set.

    Args:
        adv_model: the adversarial model M_adv. Default: deepcopy of target.
        seed_prompts: harmful prompts to attack. Default: contamination set.
        safety_reward_fn: ``(prompt, response) -> float in [0, 1]``, higher =
            safer. Default: refusal-prefix match.
        helpfulness_reward_fn: optional utility guard.
        num_rounds / num_candidates / safety_threshold / adv_steps /
        tgt_steps / max_new_tokens: MARTConfig overrides (paper defaults).
    """
    METHOD = "MART"

    def __init__(self, model=None, tokenizer=None, *,
                 adv_model=None,
                 seed_prompts=None,
                 safety_reward_fn=None,
                 helpfulness_reward_fn=None,
                 num_rounds: int = 4,
                 num_candidates: int = 8,
                 safety_threshold: float = 0.5,
                 adv_steps: int = 100,
                 tgt_steps: int = 200,
                 max_new_tokens: int = 256,
                 adv_batch_size: int = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.adv_model = adv_model
        self.seed_prompts = seed_prompts
        self.safety_reward_fn = safety_reward_fn
        self.helpfulness_reward_fn = helpfulness_reward_fn
        self.num_rounds = num_rounds
        self.num_candidates = num_candidates
        self.safety_threshold = safety_threshold
        self.adv_steps = adv_steps
        self.tgt_steps = tgt_steps
        self.max_new_tokens = max_new_tokens
        self.adv_batch_size = adv_batch_size or self.batch_size

    def train(self, train_dataset=None, out_dir: str = None, **kwargs) -> str:
        if train_dataset is not None:
            logger.warning(
                "MART: train_dataset is ignored — MART is driven by "
                "seed_prompts and the reward function. Pass seed_prompts= to "
                "control the attack surface."
            )
        seeds = self.seed_prompts
        if seeds is None:
            contamination, _refusal = _raw_pairs()
            seeds = [prompt for prompt, _ in contamination]
        adv = self.adv_model if self.adv_model is not None else copy.deepcopy(self.model)
        reward = self.safety_reward_fn or _refusal_reward

        from safetune.harden.mart import MARTConfig, MARTTrainer as MART
        cfg = MARTConfig(
            num_rounds=self.num_rounds,
            num_candidates=self.num_candidates,
            safety_threshold=self.safety_threshold,
            adv_lr=self.lr,
            tgt_lr=self.lr,
            adv_steps=self.adv_steps,
            tgt_steps=self.tgt_steps,
            adv_batch_size=self.adv_batch_size,
            tgt_batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            device=str(resolve_device()),
        )
        trained = MART(
            target_model=self.model,
            adv_model=adv,
            tokenizer=self.tok,
            seed_prompts=seeds,
            safety_reward_fn=reward,
            helpfulness_reward_fn=self.helpfulness_reward_fn,
            config=cfg,
        ).train()
        del adv
        free()
        out_dir = self._resolve_out_dir(out_dir)
        return self._save_merged(trained, out_dir)


# ── DeepRefusal: refusal-direction ablation + LoRA ──────────────────────────

class DeepRefusalTrainer(_HardenBase):
    """DeepRefusal: trains a refusal direction under ablation and distills it
    into LoRA weights (paper defaults: p=0.5, alpha=0.2, r=16).

    The direction is derived with ``compute_refusal_direction`` from the same
    harmful/harmless prompt contrast the steer trainers calibrate on
    (BeaverTails harmful prompts vs Alpaca instructions), so no vector file is
    needed. ``harmful_dataset`` is the refusal set (harmful prompt → refusal
    answer — what the model should learn to refuse); ``benign_dataset`` is the
    CLI's task set (what it must keep doing).

    Args:
        layer_idx: hidden layer the direction is read from. Default -1 (last).
        calib_n: prompts per side for the direction. Default: runtime calib_n.
        ablation_prob / alpha: DeepRefusalConfig hyperparameters.
        lora_r / lora_alpha / lora_dropout / target_modules: LoRA settings
            (``lora_r=0`` trains all parameters).
        merge_after_training: merge the LoRA adapter before saving.
    """
    METHOD = "DeepRefusal"
    HF_TRAINER = _DeepRefusalHF

    def __init__(self, model=None, tokenizer=None, *,
                 layer_idx: int = -1,
                 calib_n: int = None,
                 ablation_prob: float = 0.5,
                 alpha: float = 0.2,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 target_modules=None,
                 merge_after_training: bool = True,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.layer_idx = layer_idx
        self.calib_n = calib_n
        self.ablation_prob = ablation_prob
        self.alpha = alpha
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.merge_after_training = merge_after_training

    def train(self, train_dataset=None, out_dir: str = None, *,
              refusal_dataset=None, safety_dataset=None, **kwargs) -> str:
        from safetune.core.extras.subspace import compute_refusal_direction
        from safetune.runner.utils.data_utils import refusal_prompt_pairs_large

        if HARD.DeepRefusalConfig is None:
            raise ImportError("transformers is required for DeepRefusalTrainer")
        harmful, harmless = refusal_prompt_pairs_large(self.calib_n)
        direction = compute_refusal_direction(
            self.model, harmful, harmless,
            layer_idx=self.layer_idx, tokenizer=self.tok,
        )

        harmful_ds = safety_dataset or refusal_dataset
        if harmful_ds is None:
            _contamination, harmful_ds = _tokenized_splits(self.tok)
        benign_ds = train_dataset
        if benign_ds is None:
            benign_ds = _tokenized_splits(self.tok)[0]
        harmful_ds = _as_torch(harmful_ds)
        benign_ds = _as_torch(benign_ds)

        out_dir = self._resolve_out_dir(out_dir)
        cfg = HARD.DeepRefusalConfig(**_training_kwargs(
            self, out_dir,
            ablation_prob=self.ablation_prob,
            alpha=self.alpha,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            merge_after_training=self.merge_after_training,
        ))
        trainer = self.HF_TRAINER(
            model=self.model,
            args=cfg,
            processing_class=self.tok,
            refusal_direction=direction,
            harmful_dataset=harmful_ds,
            benign_dataset=benign_ds,
        )
        trainer.train()
        return self._save_merged(self.model, out_dir)


# ── Antibody: SAM flatness alignment + likelihood-ratio reweighting ─────────

class AntibodyTrainer(_HardenBase):
    """Antibody's two-stage defense: (1) SAM flatness alignment on harmful
    batches with a refusal anchor, (2) likelihood-ratio reweighting against a
    simulated adversary. ``mode`` defaults to ``"both"`` — the method is the
    two stages.

    ``train_dataset`` is the benign task set the model must keep; the harmful
    and refusal iterators default to the harden contamination / refusal sets.

    Args:
        mode: ``"align"``, ``"finetune"`` or ``"both"`` (default).
        refresh_every: steps between fresh harmful/refusal draws.
        sam_rho: SAM perturbation radius. Default 0.05.
        sim_threshold: similarity threshold for the simulated adversary.
        lambda_refusal / lambda_cap / xi / tau: Antibody loss weights.
    """
    METHOD = "Antibody"
    HF_TRAINER = _AntibodyHF

    def __init__(self, model=None, tokenizer=None, *,
                 mode: str = "both",
                 refresh_every: int = 1,
                 sam_rho: float = 0.05,
                 sim_threshold: float = 0.0,
                 lambda_refusal: float = 1.0,
                 lambda_cap: float = 10.0,
                 xi: float = 0.0,
                 tau: float = 1.0,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.mode = mode
        self.refresh_every = refresh_every
        self.sam_rho = sam_rho
        self.sim_threshold = sim_threshold
        self.lambda_refusal = lambda_refusal
        self.lambda_cap = lambda_cap
        self.xi = xi
        self.tau = tau

    def train(self, train_dataset=None, out_dir: str = None, *,
              harmful_dataset=None, refusal_dataset=None, safety_dataset=None,
              **kwargs) -> str:
        contamination, refusal_aux = _tokenized_splits(self.tok)
        if HARD.AntibodyConfig is None:
            raise ImportError("transformers is required for AntibodyTrainer")
        harmful_ds = _as_torch(harmful_dataset or safety_dataset or contamination)
        refusal_ds = _as_torch(refusal_dataset or safety_dataset or refusal_aux)

        out_dir = self._resolve_out_dir(out_dir)
        modes = ["align", "finetune"] if self.mode in ("both", None) else [self.mode]
        for stage in modes:
            cfg = HARD.AntibodyConfig(**_training_kwargs(
                self, out_dir,
                sam_rho=self.sam_rho,
                sim_threshold=self.sim_threshold,
            ))
            trainer = self.HF_TRAINER(
                model=self.model,
                args=cfg,
                train_dataset=train_dataset,
                processing_class=self.tok,
                harmful_dataset=harmful_ds,
                refusal_dataset=refusal_ds,
                mode=stage,
                refresh_every=self.refresh_every,
                sam_rho=self.sam_rho,
                lambda_refusal=self.lambda_refusal,
                lambda_cap=self.lambda_cap,
                xi=self.xi,
                tau=self.tau,
            )
            trainer.train()
            del trainer
            free()
        return self._save_merged(self.model, out_dir)
