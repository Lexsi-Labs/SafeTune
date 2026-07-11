"""Harden runner — tamper-resistant trainers (DOOR, RepNoise, CTRAP, SEAM)."""
import logging
import os
import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

from ._base import (
    _HardenBase, _keep_model_columns, default_data_collator, _BFLOAT16,
)
from safetune.runner.utils.model_utils import free
import safetune.harden as HARD
from torch.utils.data import DataLoader

# ── DOORTrainer ───────────────────────────────────────────────────────────────

class DOORTrainer(_HardenBase):
    """DOOR: NPO negative preference + refusal reward joint training.

    Args:
        beta: DPO/NPO temperature. Default 0.5.
        refusal_w: refusal reward loss weight. Default 1.0.
        unlearn_w: NPO unlearn loss weight. Default 1.0.
        base_model_path: reference model path for NPO log-ratio.
    """

    METHOD = "DOOR"

    def __init__(self, model=None, tokenizer=None, *,
                 beta: float = 0.5,
                 refusal_w: float = 1.0,
                 unlearn_w: float = 1.0,
                 base_model_path: str = None,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.beta = beta
        self.refusal_w = refusal_w
        self.unlearn_w = unlearn_w
        self.base_model_path = base_model_path

    def train(self, train_dataset, out_dir: str = None, *,
              contamination_pairs=None, refusal_pairs=None, **kwargs) -> str:
        import torch.nn.functional as F
        from safetune.harden.door import _get_sequence_log_probs

        # DOOR trains purely on refusal / contamination pairs (NPO unlearning +
        # refusal reward); it has no task-SFT term, so a user-supplied
        # ``train_dataset`` is not incorporated. Warn rather than silently
        # discard it so the caller isn't misled into thinking their task data
        # shaped the run.
        if train_dataset is not None:
            logger.warning(
                "DOORTrainer: train_dataset is ignored — DOOR trains only on "
                "refusal/contamination pairs (NPO unlearning + refusal reward) "
                "and has no task-SFT term. Pass contamination_pairs/refusal_pairs "
                "to control the training data."
            )

        if contamination_pairs is None or refusal_pairs is None:
            from safetune.runner.utils.data_utils import harden_contamination_pairs
            _cont, _, _ref = harden_contamination_pairs(256)
            contamination_pairs = _cont
            refusal_pairs = _ref

        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        dev = next(model.parameters()).device
        ref_path = self.base_model_path or getattr(self.tok, "name_or_path", None)
        # Load the reference on the SAME device/dtype as the model — device-agnostic
        # (CUDA / Apple-Silicon MPS / CPU) rather than forcing a single device.
        ref = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=next(model.parameters()).dtype).to(dev).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        def _tok_pairs(pairs):
            from safetune.runner.utils.data_utils import _chat_text
            ids, masks, all_labels = [], [], []
            for u, a in pairs:
                enc = self.tok(_chat_text(self.tok, u, a), truncation=True,
                               max_length=256, padding="max_length",
                               return_tensors="pt")
                input_ids = enc["input_ids"][0]
                attn = enc["attention_mask"][0]
                # Mask the prompt span (as the sibling DeRTa / data_utils
                # tokenizers do) so the refusal CE and NPO log-probs are taken
                # over the *response* tokens only, not the shared prompt.
                prompt_text = self.tok.apply_chat_template(
                    [{"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True)
                prompt_len = len(self.tok(prompt_text, truncation=True,
                                          max_length=256)["input_ids"])
                labels = input_ids.clone()
                labels[attn == 0] = -100
                labels[:prompt_len] = -100
                ids.append(input_ids)
                masks.append(attn)
                all_labels.append(labels)
            return torch.stack(ids), torch.stack(masks), torch.stack(all_labels)

        ref_ids, ref_mask, ref_labels = _tok_pairs(refusal_pairs)
        harm_ids, harm_mask, harm_labels = _tok_pairs(contamination_pairs)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        model.train()
        n_ref, n_harm = ref_ids.shape[0], harm_ids.shape[0]
        bsz = self.batch_size
        for _ in range(self.epochs):
            for s in range(0, max(n_ref, n_harm), bsz):
                ri = [(s + j) % n_ref for j in range(bsz)]
                hi = [(s + j) % n_harm for j in range(bsz)]
                r_logits = model(ref_ids[ri].to(dev),
                                 attention_mask=ref_mask[ri].to(dev)).logits
                refusal = -_get_sequence_log_probs(
                    r_logits, ref_labels[ri].to(dev)).mean()
                h_ids = harm_ids[hi].to(dev)
                h_mask = harm_mask[hi].to(dev)
                h_labels = harm_labels[hi].to(dev)
                logp_main = _get_sequence_log_probs(
                    model(h_ids, attention_mask=h_mask).logits, h_labels)
                with torch.no_grad():
                    logp_ref = _get_sequence_log_probs(
                        ref(h_ids, attention_mask=h_mask).logits, h_labels)
                npo = (-F.logsigmoid(-self.beta * (logp_main - logp_ref))
                       * (2.0 / self.beta)).mean()
                loss = self.refusal_w * refusal + self.unlearn_w * npo
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        del ref; free()
        return self._save_merged(model, out_dir)

# ── RepNoiseTrainer ───────────────────────────────────────────────────────────

class RepNoiseTrainer(_HardenBase):
    """RepNoise: representation-noising defence (Rosati et al., arXiv:2405.14577).

    Loss (reference ``rep_noise_loss``):
    ``L = L_retain + beta*L_noise(MMD) - alpha*log(L_harmful)``.

    Args:
        noise_alpha: ascent weight ``alpha`` on harmful CE -> ``repnoise_beta1``.
            Default 1.0.
        noise_beta: MMD-noise weight ``beta`` -> ``repnoise_beta3``. Default 0.1.
        noise_gamma: retain (benign CE) weight -> ``repnoise_beta2``. Default 1.0.
    """

    METHOD = "RepNoise"

    def __init__(self, model=None, tokenizer=None, *,
                 noise_alpha: float = 1.0,
                 noise_beta: float = 0.1,
                 noise_gamma: float = 1.0,
                 **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.noise_alpha = noise_alpha
        self.noise_beta = noise_beta
        self.noise_gamma = noise_gamma

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, harmful_dataset=None, **kwargs) -> str:
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)
        # RepNoise needs actual harmful (unsafe) data for the noise/ascent terms;
        # using the safety dataset here would invert the gradient ascent direction.
        if harmful_dataset is None:
            from safetune.runner.utils.data_utils import harden_contamination_sets
            harmful_dataset = harden_contamination_sets(self.tok, n=256)[0]
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(
            HARD.RepNoiseConfig(repnoise_beta1=self.noise_alpha,
                                repnoise_beta3=self.noise_beta,
                                repnoise_beta2=self.noise_gamma),
            out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(harmful_dataset, "with_format"):
            harmful_dataset = harmful_dataset.with_format("torch")

        harmful_loader = DataLoader(
            _keep_model_columns(harmful_dataset), batch_size=self.batch_size, shuffle=True,
            collate_fn=default_data_collator)
        tr = HARD.RepNoiseTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            harmful_dataset=harmful_loader)
        tr.train()
        return self._save_merged(model, out_dir)

# ── CTRAPTrainer ──────────────────────────────────────────────────────────────

class CTRAPTrainer(_HardenBase):
    """CTRAP: embedding collapse trap for safety."""

    METHOD = "CTRAP"

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, harmful_dataset=None, **kwargs) -> str:
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)
        # CTRAP simulates θ' = θ - α·∇_θ L(θ; D_harmful) to prime the collapse
        # trap; safety data here would make the trap fire on benign fine-tuning.
        if harmful_dataset is None:
            from safetune.runner.utils.data_utils import harden_contamination_sets
            harmful_dataset = harden_contamination_sets(self.tok, n=256)[0]
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(HARD.CTRAPConfig(), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(harmful_dataset, "with_format"):
            harmful_dataset = harmful_dataset.with_format("torch")
        tr = HARD.CTRAPTrainer(
            model=model, args=args,
            train_dataset=_keep_model_columns(train_dataset),
            data_collator=default_data_collator,
            harmful_dataset=_keep_model_columns(harmful_dataset))
        tr.train()
        return self._save_merged(model, out_dir)

# ── SEAMTrainer ───────────────────────────────────────────────────────────────

class SEAMTrainer(_HardenBase):
    """SEAM: semantic alignment during fine-tuning."""

    METHOD = "SEAM"

    def train(self, train_dataset, out_dir: str = None, *,
              safety_dataset=None, harmful_dataset=None, **kwargs) -> str:
        if safety_dataset is None:
            from safetune.runner.utils.data_utils import build_safety_dataset
            safety_dataset = build_safety_dataset(self.tok)
        # SEAM (arXiv:2505.12186): D_bgn=train_dataset, D_adv=harmful set,
        # D_aln=refusal/utility set. D_adv must be harmful data so that SEAM
        # unlearns the right distribution; safety data in that slot would have
        # SEAM unlearn refusal behavior instead.
        if harmful_dataset is None:
            from safetune.runner.utils.data_utils import harden_contamination_sets
            harmful_dataset = harden_contamination_sets(self.tok, n=256)[0]
        out_dir = self._resolve_out_dir(out_dir)
        model = self._lora_base()
        args = self._configure_args(HARD.SEAMConfig(), out_dir)
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if hasattr(safety_dataset, "with_format"):
            safety_dataset = safety_dataset.with_format("torch")
        if hasattr(harmful_dataset, "with_format"):
            harmful_dataset = harmful_dataset.with_format("torch")
        bden = _keep_model_columns(train_dataset)
        safe = _keep_model_columns(safety_dataset)
        harm = _keep_model_columns(harmful_dataset)
        tr = HARD.SEAMTrainer(
            model=model, args=args,
            train_dataset=bden,
            data_collator=default_data_collator,
            harmful_dataset=harm,
            alignment_dataset=safe,
            benign_dataset=bden)
        tr.train()
        return self._save_merged(model, out_dir)

