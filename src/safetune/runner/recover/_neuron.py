"""Recover runner — neuron trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── NLSRTrainer ───────────────────────────────────────────────────────────────

class NLSRTrainer(_RecoverBase):
    """NLSR: Neuron-Level Safety Restoration via donor state transplant.

    Args:
        donor_state: state dict from the aligned model.
        blend: interpolation coefficient. Default 0.5.
    """

    METHOD = "NLSRTrainer"

    def __init__(self, model=None, *, donor_state=None, blend: float = 0.5, **kwargs):
        super().__init__(model, **kwargs)
        self.donor_state = donor_state
        self.blend = blend

    def apply(self, *, blend: float = None, **kwargs):
        return R.apply_nlsr(
            self.model,
            donor_state=self.donor_state,
            blend=blend if blend is not None else self.blend,
        )

# ── AntidoteTrainer ───────────────────────────────────────────────────────────

class AntidoteTrainer(_RecoverBase):
    """Antidote: WANDA-style harmful neuron pruning.

    Args:
        prune_fraction: fraction of weights to prune. Default 0.005.
        harmful_prompts: list of harmful calibration prompts.
        tokenizer: tokenizer for the model.
        max_samples: max calibration samples. Default 64.
    """

    METHOD = "AntidoteTrainer"

    def __init__(self, model=None, *, prune_fraction: float = 0.005,
                 harmful_prompts=None, tokenizer=None, max_samples: int = 64,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.prune_fraction = prune_fraction
        self.harmful_prompts = harmful_prompts
        self.tokenizer = tokenizer
        self.max_samples = max_samples

    def apply(self, *, harmful_prompts=None, tokenizer=None,
              prune_fraction: float = None, **kwargs):
        hp = harmful_prompts if harmful_prompts is not None else self.harmful_prompts
        if hp is None:
            from safetune.runner.utils.data_utils import refusal_prompt_pairs_large
            hp, _ = refusal_prompt_pairs_large(self.max_samples)
        return R.apply_antidote(
            self.model,
            prune_fraction=prune_fraction if prune_fraction is not None
            else self.prune_fraction,
            harmful_prompts=hp,
            tokenizer=tokenizer or self.tokenizer,
            max_samples=self.max_samples,
        )

# ── MSCPTrainer ───────────────────────────────────────────────────────────────

class MSCPTrainer(_RecoverBase):
    """MSCP: Model Safety Constraint Projection.

    Args:
        aligned_state: state dict of the aligned model.
        base_state: state dict of the base model.
        coefficient: projection coefficient. Default 0.05.
    """

    METHOD = "MSCPTrainer"

    def __init__(self, model=None, *, aligned_state=None, base_state=None,
                 coefficient: float = 0.05, **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_state = aligned_state
        self.base_state = base_state
        self.coefficient = coefficient

    def apply(self, *, coefficient: float = None, **kwargs):
        return R.apply_mscp(
            self.model,
            aligned_state=self.aligned_state,
            base_state=self.base_state,
            coefficient=coefficient if coefficient is not None else self.coefficient,
        )

# ── AntidoteV2Trainer ─────────────────────────────────────────────────────────

class AntidoteV2Trainer(_RecoverBase):
    """Antidote v2: WANDA pruning with utility floor and overlap budget.

    Args:
        tokenizer: tokenizer.
        harmful_prompts: harmful calibration prompts.
        benign_prompts: benign calibration prompts.
        global_prune_fraction: global pruning fraction. Default 0.005.
        utility_floor: minimum utility retention. Default 0.1.
        overlap_budget: harm/utility overlap budget. Default 0.05.
        max_samples: max calibration samples. Default 64.
    """

    METHOD = "AntidoteV2Trainer"

    def __init__(self, model=None, *, tokenizer=None, harmful_prompts=None,
                 benign_prompts=None, global_prune_fraction: float = 0.005,
                 utility_floor: float = 0.1, overlap_budget: float = 0.05,
                 max_samples: int = 64, **kwargs):
        super().__init__(model, **kwargs)
        self.tokenizer = tokenizer
        self.harmful_prompts = harmful_prompts
        self.benign_prompts = benign_prompts
        self.global_prune_fraction = global_prune_fraction
        self.utility_floor = utility_floor
        self.overlap_budget = overlap_budget
        self.max_samples = max_samples

    def apply(self, *, harmful_prompts=None, benign_prompts=None,
              tokenizer=None, **kwargs):
        hp = harmful_prompts or self.harmful_prompts
        bp = benign_prompts or self.benign_prompts
        if hp is None or bp is None:
            from safetune.runner.utils.data_utils import refusal_prompt_pairs_large
            _hp, _bp = refusal_prompt_pairs_large(self.max_samples)
            hp = hp or _hp
            bp = bp or _bp
        return R.apply_antidote_v2(
            self.model,
            tokenizer=tokenizer or self.tokenizer,
            harmful_prompts=hp,
            benign_prompts=bp,
            global_prune_fraction=self.global_prune_fraction,
            utility_floor=self.utility_floor,
            overlap_budget=self.overlap_budget,
            max_samples=self.max_samples,
        )

