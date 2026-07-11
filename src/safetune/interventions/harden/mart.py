"""
MART: Multi-round Automatic Red-Teaming — faithful standalone implementation.

Faithful implementation of the MART algorithm from:

    "Improving LLM Safety with Multi-round Automatic Red-Teaming"
    Ge, Zhou, Liu, Sun, Xie, Wang, Yang, Xu. arXiv:2311.07689.

MART co-evolves two LLMs over T rounds (paper: T=4):

  * **M_adv** — an adversarial LLM that learns to generate harder and harder
    attack prompts via SFT on successful attacks found each round.
  * **M_tgt** — the target LLM being hardened; each round it sees harder
    attacks and is SFT-trained to respond safely (and helpfully) to them.

Two reward models gate the target update:

  * A **safety reward model** scores each candidate response from M_tgt;
    only responses that clear ``safety_threshold`` are eligible.
  * An optional **helpfulness reward model** ensures the selected response
    is also helpful; when provided, the final selection maximises
    ``safety_score + helpfulness_score``.

Relationship to ASRTCallback
------------------------------
:class:`~safetune.harden.asrt.ASRTCallback` is a SafeTune-original heuristic
that wraps any HuggingFace Trainer with a *single*-model periodic red-team
loop. It is **not** MART. MART requires two co-evolving, separately-trained
LLMs and reward-model-filtered candidate selection — none of which fit the
``TrainerCallback`` model. :class:`MARTTrainer` is the faithful MART
orchestrator; :class:`~safetune.harden.asrt.ASRTCallback` is the simpler
single-model callback.

Usage::

    from safetune.harden.mart import MARTTrainer, MARTConfig

    trainer = MARTTrainer(
        target_model=model,
        adv_model=adv_model,           # second LLM; initially same weights
        tokenizer=tokenizer,
        seed_prompts=harmful_prompts,  # list[str] — bootstrap pool
        safety_reward_fn=safety_rm,    # callable(prompt, response) -> float
        helpfulness_reward_fn=help_rm, # optional
        config=MARTConfig(),
    )
    hardened_model = trainer.train()

Reward function interface
--------------------------
``safety_reward_fn(prompt: str, response: str) -> float`` where 0.0 = unsafe
and 1.0 = safe.  This is a plain Python callable; any SafeTune
``HFJudge`` / ``StringMatchJudge`` can be wrapped trivially.  The same
signature applies to ``helpfulness_reward_fn``.

Memory
-------
Both models are moved to GPU only during their respective generate / fine-tune
phases and returned to CPU between rounds, keeping GPU memory proportional to
a single model at a time.  Set ``config.device`` to override the device used
for active phases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heavy imports — guarded so the module is importable even without torch /
# transformers (mirrors the convention in door.py, derta.py, etc.).
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    AdamW = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = _e

try:
    from torch.utils.data import DataLoader, TensorDataset
    _DATALOADER_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _DATALOADER_IMPORT_ERROR = _e

try:
    from transformers import PreTrainedTokenizerBase
    _TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    PreTrainedTokenizerBase = object  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = _e


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MARTConfig:
    """Hyper-parameters for the MART co-evolution loop.

    Attributes:
        num_rounds: Number of outer MART rounds (paper: T=4).
        num_candidates: Candidate responses sampled from M_tgt per attack
            prompt during Phase B (the paper uses best-of-N sampling).
        safety_threshold: Minimum safety score a candidate response must
            achieve to be considered at all.  Responses below this score
            are never used for M_tgt fine-tuning.
        adv_lr: AdamW learning rate for M_adv SFT updates.
        tgt_lr: AdamW learning rate for M_tgt SFT updates.
        adv_steps: Gradient steps per M_adv SFT update.
        tgt_steps: Gradient steps per M_tgt SFT update.
        adv_batch_size: Per-step batch size for M_adv SFT.
        tgt_batch_size: Per-step batch size for M_tgt SFT.
        max_new_tokens: Maximum number of new tokens for both generators.
        device: Torch device string used for active (generate / train) phases.
            ``"auto"`` resolves to ``"cuda"`` when available, else ``"cpu"``.
        max_prompt_length: Token budget for prompts fed to both models.
            Longer prompts are truncated on the left.
    """

    num_rounds: int = 4
    num_candidates: int = 8
    safety_threshold: float = 0.5
    adv_lr: float = 2e-5
    tgt_lr: float = 2e-5
    adv_steps: int = 100
    tgt_steps: int = 200
    adv_batch_size: int = 4
    tgt_batch_size: int = 4
    max_new_tokens: int = 256
    device: str = "auto"
    max_prompt_length: int = 512


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MARTTrainer:
    """MART orchestrator: manages M_adv and M_tgt co-evolution.

    This is the faithful MART implementation (Ge et al., arXiv:2311.07689).
    For the simpler single-model callback see
    :class:`~safetune.harden.asrt.ASRTCallback`.

    Each of the ``num_rounds`` outer iterations runs:

    * **Phase A** — update M_adv: generate completions for the current attack
      pool, identify prompts that successfully elicited harmful responses from
      M_tgt, and SFT M_adv on those successful prompts so it learns to
      generate better attacks.
    * **Phase B** — update M_tgt: let the freshly updated M_adv generate a
      new (harder) attack pool; for each attack prompt sample
      ``num_candidates`` responses from M_tgt; pick the one that scores
      highest on safety (and optionally helpfulness) with the reward models;
      SFT M_tgt on (attack_prompt, best_safe_response) pairs.

    Round 0 bootstraps with ``seed_prompts`` as the Phase B attack pool.
    After each round, the pool is augmented with prompts where M_tgt was
    successfully attacked (i.e. produced an unsafe response).

    Args:
        target_model: The LLM being hardened (M_tgt).  Must be a
            ``torch.nn.Module`` with a HuggingFace-compatible forward
            signature.
        adv_model: The adversarial LLM (M_adv).  Initially loaded with the
            same weights; co-trained alongside ``target_model``.
        tokenizer: Shared tokenizer for both models.
        seed_prompts: List of known harmful prompts used to bootstrap round 0.
        safety_reward_fn: ``(prompt, response) -> float`` in [0, 1];
            1.0 = fully safe.
        helpfulness_reward_fn: Optional ``(prompt, response) -> float`` in
            [0, 1]; 1.0 = maximally helpful.  When ``None``, helpfulness
            filtering is skipped.
        config: :class:`MARTConfig`; paper defaults when omitted.
    """

    def __init__(
        self,
        target_model: "nn.Module",
        adv_model: "nn.Module",
        tokenizer: "PreTrainedTokenizerBase",
        seed_prompts: List[str],
        safety_reward_fn: Callable[[str, str], float],
        helpfulness_reward_fn: Optional[Callable[[str, str], float]] = None,
        config: Optional[MARTConfig] = None,
    ) -> None:
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError("torch is required for MARTTrainer") from _TORCH_IMPORT_ERROR
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise ImportError(
                "transformers is required for MARTTrainer"
            ) from _TRANSFORMERS_IMPORT_ERROR
        if not seed_prompts:
            raise ValueError("MARTTrainer: seed_prompts cannot be empty.")

        self.target_model: nn.Module = target_model
        self.adv_model: nn.Module = adv_model
        self.tokenizer = tokenizer
        self.seed_prompts: List[str] = list(seed_prompts)
        self.safety_reward_fn = safety_reward_fn
        self.helpfulness_reward_fn = helpfulness_reward_fn
        self.config = config or MARTConfig()

        # Resolve compute device.
        if self.config.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.config.device)

        # Ensure the tokenizer has a pad token so left-padding works.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Mutable attack prompt pool — bootstrapped with seed prompts.
        self._attack_pool: List[str] = list(seed_prompts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> "nn.Module":
        """Run all MART rounds. Returns the trained target model.

        Round 0 uses ``seed_prompts`` as the initial Phase B attack pool.
        From round 1 onward, Phase A first updates M_adv (using the attack
        pool grown by Phase B of the previous round), then Phase B uses the
        freshly updated M_adv to generate a new, harder attack batch.
        """
        for round_idx in range(self.config.num_rounds):
            logger.info("MART: starting round %d/%d", round_idx + 1, self.config.num_rounds)
            self._update_adv(round_idx)
            self._update_tgt(round_idx)
            logger.info(
                "MART: round %d complete; attack pool size = %d",
                round_idx + 1,
                len(self._attack_pool),
            )
        return self.target_model

    # ------------------------------------------------------------------
    # Phase A: update adversarial model
    # ------------------------------------------------------------------

    def _update_adv(self, round_idx: int) -> None:
        """Phase A — SFT M_adv on successful attack prompts.

        Steps:
        1. Generate completions for the current attack pool using M_tgt.
        2. Score each completion with the safety reward; completions with
           score < safety_threshold are deemed "successful attacks" (M_tgt
           failed to refuse).
        3. SFT M_adv on the successful prompts so it learns the distribution
           of attacks that work.

        In round 0 there is no prior M_tgt history, so we still collect
        prompts from the seed pool.  The adversarial model learns from
        whatever attacks the seed pool contains.
        """
        logger.info("MART round %d: Phase A — updating M_adv", round_idx + 1)
        pool = self._attack_pool
        if not pool:
            logger.warning("MART: attack pool is empty; skipping Phase A.")
            return

        # 1. Generate M_tgt completions for the current pool.
        tgt_responses = self._generate(self.target_model, pool, n=1)
        # tgt_responses is a flat list of len(pool) strings (n=1).

        # 2. Identify successful attacks (M_tgt gave unsafe response).
        successful_prompts: List[str] = []
        for prompt, response in zip(pool, tgt_responses):
            try:
                score = float(self.safety_reward_fn(prompt, response))
            except Exception as exc:  # pragma: no cover
                logger.warning("MART: safety_reward_fn raised %s; treating as safe.", exc)
                score = 1.0
            if score < self.config.safety_threshold:
                # Low safety score -> the attack was successful.
                successful_prompts.append(prompt)

        logger.info(
            "MART round %d Phase A: %d/%d prompts were successful attacks.",
            round_idx + 1,
            len(successful_prompts),
            len(pool),
        )

        if not successful_prompts:
            logger.info(
                "MART round %d Phase A: no successful attacks; M_adv not updated.", round_idx + 1
            )
            return

        # 3. SFT M_adv on successful prompts (it learns to produce them).
        # The training signal is: given the user instruction prefix, reproduce
        # the attack prompt.  We treat each attack prompt as both the context
        # and the target (unconditional LM on attack strings, which is the
        # paper's "SFT M_adv on successful attacks" step).
        adv_data = self._encode_texts(successful_prompts)
        self._sft_step(
            model=self.adv_model,
            data=adv_data,
            lr=self.config.adv_lr,
            steps=self.config.adv_steps,
            batch_size=self.config.adv_batch_size,
        )

    # ------------------------------------------------------------------
    # Phase B: update target model
    # ------------------------------------------------------------------

    def _update_tgt(self, round_idx: int) -> None:
        """Phase B — SFT M_tgt on (attack_prompt, best_safe_response) pairs.

        Steps:
        1. Use the updated M_adv to generate new (harder) attack prompts,
           conditioning on each existing pool entry as a prefix.
        2. For each attack prompt, sample ``num_candidates`` responses from
           M_tgt.
        3. Score responses with the safety (and helpfulness) reward models.
        4. Select the candidate with the highest combined score, provided it
           clears ``safety_threshold``.
        5. SFT M_tgt on the (attack_prompt, best_safe_response) pairs.
        6. Extend the attack pool with prompts where no safe-enough candidate
           was found (M_tgt was "successfully attacked" by all candidates).
        """
        logger.info("MART round %d: Phase B — updating M_tgt", round_idx + 1)
        pool = self._attack_pool
        if not pool:
            logger.warning("MART: attack pool is empty; skipping Phase B.")
            return

        # 1. Generate new attack prompts from M_adv (one per pool entry).
        new_attack_prompts = self._generate(self.adv_model, pool, n=1)

        # 2 & 3. For each new attack prompt, sample candidates from M_tgt and
        #        score them.
        sft_pairs: List[tuple] = []   # (attack_prompt, best_response)
        newly_failed: List[str] = []  # prompts where no safe response found

        for attack_prompt, responses in zip(
            new_attack_prompts,
            self._generate_batched_candidates(self.target_model, new_attack_prompts),
        ):
            best_response, best_score = self._select_best_response(attack_prompt, responses)
            if best_response is not None:
                sft_pairs.append((attack_prompt, best_response))
            else:
                # M_tgt couldn't produce a safe response -> add to attack pool.
                newly_failed.append(attack_prompt)

        logger.info(
            "MART round %d Phase B: %d training pairs collected; "
            "%d prompts added to attack pool.",
            round_idx + 1,
            len(sft_pairs),
            len(newly_failed),
        )

        # 6. Grow the attack pool with hard prompts M_tgt failed on.
        self._attack_pool.extend(newly_failed)

        if not sft_pairs:
            logger.info(
                "MART round %d Phase B: no safe candidates found; M_tgt not updated.",
                round_idx + 1,
            )
            return

        # 5. SFT M_tgt on (attack_prompt, best_safe_response) pairs.
        # We train on the *concatenation* of prompt + response so that the
        # model sees the full context.  The prompt tokens are masked out of
        # the loss (label = -100) so only the response is supervised.
        tgt_data = self._encode_pairs(sft_pairs)
        self._sft_step(
            model=self.target_model,
            data=tgt_data,
            lr=self.config.tgt_lr,
            steps=self.config.tgt_steps,
            batch_size=self.config.tgt_batch_size,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate(
        self,
        model: "nn.Module",
        prompts: List[str],
        n: int = 1,
    ) -> List[str]:
        """Generate ``n`` completions per prompt.

        When ``n == 1`` returns a flat list of ``len(prompts)`` strings.
        When ``n > 1`` returns a flat list of ``len(prompts) * n`` strings
        (all candidates for prompt[0], then all for prompt[1], …).

        The model is moved to ``self._device`` for generation and back to
        CPU afterward to keep memory usage bounded.
        """
        if not prompts:
            return []

        was_training = model.training
        model.eval()
        device = self._device
        model_device = next(model.parameters()).device
        if model_device != device:
            model.to(device)

        results: List[str] = []
        # Left-pad for batched generation: with the default right padding,
        # trailing pad tokens would sit between the prompt and the first
        # generated token, so every sequence would be conditioned on pads.
        # Save/restore so the caller's tokenizer is not left mutated.
        _has_padding_side = hasattr(self.tokenizer, "padding_side")
        if _has_padding_side:
            saved_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
        try:
            # Tokenise all prompts as a left-padded batch.
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    num_return_sequences=n,
                    do_sample=(n > 1),
                    temperature=1.0 if n > 1 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # output_ids shape: (batch * n, prompt_len + new_tokens)
            # Decode only the newly generated tokens.
            prompt_len = input_ids.shape[1]
            new_ids = output_ids[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            results = [text.strip() for text in decoded]
        except Exception as exc:  # pragma: no cover
            logger.warning("MART: generation failed (%s); returning empty strings.", exc)
            results = [""] * (len(prompts) * n)
        finally:
            # Restore the caller's tokenizer padding side.
            if _has_padding_side:
                self.tokenizer.padding_side = saved_padding_side
            # Move back to CPU to free GPU memory between phases.
            if model_device != device:
                model.to(model_device)
            elif str(device) != "cpu":
                model.to("cpu")
            if was_training:
                model.train()

        return results

    def _generate_batched_candidates(
        self,
        model: "nn.Module",
        prompts: List[str],
    ) -> List[List[str]]:
        """Generate ``num_candidates`` responses per prompt.

        Returns a list of lists: ``result[i]`` is the list of
        ``num_candidates`` candidate responses for ``prompts[i]``.
        """
        n = self.config.num_candidates
        flat = self._generate(model, prompts, n=n)
        # Reshape: flat is [p0_c0, p0_c1, …, p0_c(n-1), p1_c0, …]
        return [flat[i * n: (i + 1) * n] for i in range(len(prompts))]

    # ------------------------------------------------------------------
    # Response selection
    # ------------------------------------------------------------------

    def _select_best_response(
        self,
        prompt: str,
        candidates: List[str],
    ) -> tuple:
        """Pick the best candidate response for a given attack prompt.

        Scores each candidate with the safety reward model (and optionally
        the helpfulness reward model).  The candidate with the highest
        ``safety_score + helpfulness_score`` is selected, provided its
        safety score clears ``safety_threshold``.

        Returns:
            ``(best_response, best_score)`` if a qualifying candidate is
            found, or ``(None, 0.0)`` when none clears the threshold.
        """
        best_response: Optional[str] = None
        best_score: float = -1.0

        for candidate in candidates:
            if not candidate:
                continue
            try:
                s_score = float(self.safety_reward_fn(prompt, candidate))
            except Exception as exc:  # pragma: no cover
                logger.warning("MART: safety_reward_fn raised %s; skipping candidate.", exc)
                continue

            if s_score < self.config.safety_threshold:
                # Candidate is not safe enough; skip regardless of helpfulness.
                continue

            h_score = 0.0
            if self.helpfulness_reward_fn is not None:
                try:
                    h_score = float(self.helpfulness_reward_fn(prompt, candidate))
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "MART: helpfulness_reward_fn raised %s; helpfulness = 0.", exc
                    )

            combined = s_score + h_score
            if combined > best_score:
                best_score = combined
                best_response = candidate

        return best_response, max(best_score, 0.0)

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _encode_texts(self, texts: List[str]) -> "TensorDataset":
        """Encode a list of plain texts into a TensorDataset for SFT.

        Each text is treated as both context and label (unsupervised /
        language-model style), matching the Phase A M_adv training signal.
        """
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_new_tokens,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # Labels = input_ids; masked where padding.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return TensorDataset(input_ids, attention_mask, labels)

    def _encode_pairs(self, pairs: List[tuple]) -> "TensorDataset":
        """Encode (prompt, response) pairs with prompt tokens masked in labels.

        The concatenated ``prompt + response`` is tokenised.  Label positions
        corresponding to the prompt prefix are set to ``-100`` so the SFT
        loss only supervises the response tokens.
        """
        all_input_ids: List["torch.Tensor"] = []
        all_attention_mask: List["torch.Tensor"] = []
        all_labels: List["torch.Tensor"] = []

        max_total = self.config.max_prompt_length + self.config.max_new_tokens

        for prompt, response in pairs:
            # Tokenise prompt and response separately to get the boundary.
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_prompt_length,
            )
            response_ids = self.tokenizer.encode(
                response,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_new_tokens,
            )
            # Append EOS if not already present.
            eos = self.tokenizer.eos_token_id
            if eos is not None and (not response_ids or response_ids[-1] != eos):
                response_ids.append(eos)

            combined = prompt_ids + response_ids
            if len(combined) > max_total:
                combined = combined[:max_total]

            input_ids = torch.tensor(combined, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            # Mask the prompt portion out of the loss.
            prompt_end = min(len(prompt_ids), len(combined))
            labels[:prompt_end] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        # Pad to the same length within this mini-batch.
        max_len = max(t.size(0) for t in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        def pad_seq(seq: "torch.Tensor", pad_value: int) -> "torch.Tensor":
            diff = max_len - seq.size(0)
            if diff == 0:
                return seq
            return torch.cat([seq, seq.new_full((diff,), pad_value)])

        padded_ids = torch.stack([pad_seq(t, pad_id) for t in all_input_ids])
        padded_mask = torch.stack([pad_seq(t, 0) for t in all_attention_mask])
        padded_labels = torch.stack([pad_seq(t, -100) for t in all_labels])

        return TensorDataset(padded_ids, padded_mask, padded_labels)

    # ------------------------------------------------------------------
    # SFT training loop
    # ------------------------------------------------------------------

    def _sft_step(
        self,
        model: "nn.Module",
        data: "TensorDataset",
        lr: float,
        steps: int,
        batch_size: int = 4,
    ) -> None:
        """Run a lightweight in-place SFT update on ``model``.

        Uses a simple AdamW loop over the provided TensorDataset.  The model
        is moved to ``self._device`` for training and back to CPU when done.

        Args:
            model: The ``nn.Module`` to update in place.
            data: A ``TensorDataset`` of ``(input_ids, attention_mask, labels)``
                tensors.  Produced by :meth:`_encode_texts` or
                :meth:`_encode_pairs`.
            lr: AdamW learning rate.
            steps: Number of gradient steps to take.  If the dataset is
                exhausted before ``steps`` steps the loader cycles
                (``drop_last=False``).
            batch_size: Number of examples per gradient step.
        """
        if len(data) == 0:  # type: ignore[arg-type]
            logger.warning("MART: _sft_step called with empty dataset; skipping.")
            return

        device = self._device
        model_device = next(model.parameters()).device
        if model_device != device:
            model.to(device)

        model.train()
        optimizer = AdamW(model.parameters(), lr=lr)

        loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
        loader_iter = iter(loader)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for step in range(steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Cycle the loader.
                loader_iter = iter(loader)
                batch = next(loader_iter)

            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # HuggingFace CausalLM models return loss directly when
                # ``labels`` are provided.
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # Fallback: compute loss from logits manually.
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1).to(shift_logits.device),
                    )
            except Exception as exc:
                logger.warning("MART: forward pass failed at step %d (%s); skipping.", step, exc)
                continue

            if not torch.isfinite(loss):
                logger.warning("MART: non-finite loss at step %d; skipping.", step)
                continue

            loss.backward()
            # Clip gradients to prevent instability with small datasets.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 10 == 0 or step == steps - 1:
                logger.debug("MART SFT step %d/%d  loss=%.4f", step + 1, steps, loss.item())

        # Release optimizer state and move model back to CPU.
        del optimizer
        if str(device) != "cpu":
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["MARTConfig", "MARTTrainer"]
