"""Recover runner — other trainer set."""
from ._base import _RecoverBase
import safetune.recover as R

# ── PKETrainer ────────────────────────────────────────────────────────────────

class PKETrainer(_RecoverBase):
    """PKE: Parameter Knowledge Editing for safety neurons.

    Args:
        clean_model: clean / aligned model.
        toxic_model: drifted / harmful model.
        top_k_neurons: number of top-k safety neurons to edit. Default 50.
        num_steps: gradient edit steps. Default 10.
    """

    METHOD = "PKETrainer"

    def __init__(self, model=None, *, clean_model=None, toxic_model=None,
                 top_k_neurons: int = 50, num_steps: int = 10, tokenizer=None,
                 harmful_prompt: str = None, safe_response: str = None,
                 locality_inputs=None, **kwargs):
        super().__init__(model, **kwargs)
        self.clean_model = clean_model
        self.toxic_model = toxic_model
        self.top_k_neurons = top_k_neurons
        self.num_steps = num_steps
        self.tokenizer = tokenizer
        self.harmful_prompt = harmful_prompt
        self.safe_response = safe_response
        self.locality_inputs = locality_inputs

    def apply(self, *, top_k_neurons: int = None, num_steps: int = None,
              tokenizer=None, harmful_prompt: str = None,
              safe_response: str = None, locality_inputs=None, **kwargs):
        # PKE's faithful DINM edit teaches the located down_proj rows to emit a
        # refusal, which needs a tokenizer; fall back to the model's tokenizer.
        tok = tokenizer or self.tokenizer
        if tok is None:
            from safetune.runner.utils.model_utils import load_tok
            tok = load_tok(self.model_id)
        opt = {}
        hp = harmful_prompt or self.harmful_prompt
        sr = safe_response or self.safe_response
        li = locality_inputs if locality_inputs is not None else self.locality_inputs
        if hp is not None:
            opt["harmful_prompt"] = hp
        if sr is not None:
            opt["safe_response"] = sr
        if li is not None:
            opt["locality_inputs"] = li
        return R.apply_pke(
            self.model,
            clean=self.clean_model,
            toxic=self.toxic_model,
            top_k_neurons=top_k_neurons if top_k_neurons is not None else self.top_k_neurons,
            num_steps=num_steps if num_steps is not None else self.num_steps,
            tokenizer=tok,
            **opt,
        )

# ── SafeReActTrainer ──────────────────────────────────────────────────────────

class SafeReActTrainer(_RecoverBase):
    """SafeReAct: reactivate dormant safety neurons via probe inputs.

    Args:
        reference_model: aligned reference model.
        probe_inputs: tokenized probe inputs for safety neurons.
        train_lora: train LoRA adapter on top. Default False.
    """

    METHOD = "SafeReActTrainer"

    def __init__(self, model=None, *, reference_model=None, probe_inputs=None,
                 train_lora: bool = False, **kwargs):
        super().__init__(model, **kwargs)
        self.reference_model = reference_model
        self.probe_inputs = probe_inputs
        self.train_lora = train_lora

    def apply(self, *, probe_inputs=None, **kwargs):
        # 1. Identify the target model's device
        target_device = next(self.model.parameters()).device
        
        # 2. Move the reference model to the target device
        if self.reference_model is not None:
            self.reference_model.to(target_device)
            
        # 3. Ensure probe_inputs are also on the correct device
        active_inputs = probe_inputs if probe_inputs is not None else self.probe_inputs
        if active_inputs is not None:
            # If inputs are a dictionary (like standard HuggingFace tokenizer output)
            if isinstance(active_inputs, dict):
                active_inputs = {k: v.to(target_device) if hasattr(v, 'to') else v for k, v in active_inputs.items()}
            # If inputs are just a standard tensor
            elif hasattr(active_inputs, 'to'):
                active_inputs = active_inputs.to(target_device)

        # 4. Run the intervention
        return R.apply_safereact(
            self.model,
            reference_model=self.reference_model,
            probe_inputs=active_inputs,
        )

# ── SCRUBTrainer ──────────────────────────────────────────────────────────────

class SCRUBTrainer(_RecoverBase):
    """SCRUB: teacher-student selective unlearning (recover variant).

    Args:
        sgda_epochs: SGDA epochs. Default 3.
        msteps: maximization steps. Default 10.
        lr: learning rate. Default 5e-5.
        max_steps: total training steps cap. Default 200.
        forget_clip: gradient clipping for forget loss. Default 0.5.
        alpha: retain loss weight. Default 1.0.
    """

    METHOD = "SCRUBTrainer"

    def __init__(self, model=None, *, sgda_epochs: int = 3, msteps: int = 10,
                 lr: float = 5e-5, max_steps: int = 200,
                 forget_clip: float = 0.5, alpha: float = 1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.sgda_epochs = sgda_epochs
        self.msteps = msteps
        self.lr = lr
        self.max_steps = max_steps
        self.forget_clip = forget_clip
        self.alpha = alpha

    def apply(self, retain=None, forget=None, **kwargs):
        from safetune.recover import SCRUBConfig
        cfg = SCRUBConfig(
            sgda_epochs=self.sgda_epochs,
            msteps=self.msteps,
            lr=self.lr,
            max_steps=self.max_steps,
            forget_clip=self.forget_clip,
            alpha=self.alpha,
        )
        return R.scrub_unlearn(self.model,
                               retain_batches=retain if retain is not None else [],
                               forget_batches=forget if forget is not None else [],
                               config=cfg)

# ── Convenience aliases ───────────────────────────────────────────────────────

__all__ = [
    "CThetaTrainer",
    "ReStaTrainer",
    "SafeLoRATrainer",
    "SafeMergeTrainer",
    "SOMFTrainer",
    "LoXTrainer",
    "TaskArithmeticTrainer",
    "SafeDeltaTrainer",
    "NLSRTrainer",
    "PKETrainer",
    "QReSafeTrainer",
    "AAQTrainer",
    "AntidoteTrainer",
    "MSCPTrainer",
    "SafeReActTrainer",
    "LSSFTrainer",
    "PrePostMergeTrainer",
    "SCRUBTrainer",
    "GradSelectiveRecoverTrainer",
    "WiseFTTrainer",
    "SafetyVectorRestoreTrainer",
    "OneShotSafetyPatchTrainer",
    "AntidoteV2Trainer",
    "RepNoiseRecoverTrainer",
    "load_recover_data",
]


def load_recover_data(model_id, max_len: int = 64):
    """Tokenized calibration inputs for recover methods. No tokenizer required."""
    import torch
    from safetune.runner.utils.dataset import load_recover_dataset
    from safetune.runner.utils.model_utils import load_tok
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_recover_dataset(load_tok(model_id), device, max_len=max_len)

