#!/usr/bin/env python3
"""SafeTune CLI: dispatch to each safety pillar."""

import argparse
import sys

from safetune.runner._registry import (
    HARDEN_REGISTRY,
    RECOVER_REGISTRY,
    UNLEARN_REGISTRY,
)


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    """Print the SafeTune terminal banner — only when stdout is a real TTY."""
    if not sys.stdout.isatty():
        return
    from safetune import __version__
    P  = "\033[38;5;99m"   # SafeTune purple
    A  = "\033[38;5;214m"  # compass-needle amber
    D  = "\033[2m"          # dim
    B  = "\033[1m"          # bold
    R  = "\033[0m"          # reset
    sep = D + "─" * 52 + R
    print(
        f"\n"
        f"  {A}◆{R}  {B}{P}S A F E T U N E{R}  {D}v{__version__} · MIT · Python 3.12+{R}\n"
        f"     {D}Fine-tuning breaks safety. SafeTune fixes it.{R}\n"
        f"  {sep}\n"
        f"  {P}harden{R} · {P}recover{R} · {P}unlearn{R} · {P}steer{R}"
        f"  {D}· interpret · evaluate{R}\n"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model_and_tok(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    # Many base/instruct tokenizers (Llama-3, Mistral) ship without a pad token.
    # Training tokenizes with padding="max_length", which raises "Asking to pad
    # but the tokenizer does not have a padding token" — mirror model_utils.load_tok
    # and fall back to EOS so the CLI works on those models out of the box.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _trainer_kwargs(args) -> dict:
    kw = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        wandb=bool(getattr(args, "wandb", False)),
    )
    # Standard YAML config fields that land on the args namespace via --config
    # parser-default injection (they have no dedicated CLI flag).
    for field in ("optimizer", "logging_steps"):
        val = getattr(args, field, None)
        if val is not None:
            kw[field] = val
    # Forward method-specific hyperparameters collected from --config
    # (e.g. lisa_rho, rank, inner_steps) — every Trainer accepts **kwargs.
    kw.update(getattr(args, "method_kwargs", None) or {})
    return kw


def _load_train_dataset(args):
    """Load the training dataset specified by --train-dataset / --train-split.

    The split default is dataset-aware: BeaverTails uses ``30k_train``; any
    other HF dataset falls back to the conventional ``train`` split, so a
    non-BeaverTails ``--train-dataset`` no longer fails on a missing
    ``30k_train`` split.
    """
    ds_name = getattr(args, "train_dataset", "beavertails") or "beavertails"
    split = getattr(args, "train_split", None)
    if ds_name == "beavertails":
        from safetune.data import load_beavertails
        return load_beavertails(split=split or "30k_train")
    from datasets import load_dataset
    return load_dataset(ds_name, split=split or "train")


def _ensure_tokenized(dataset, tok, *, max_len: int = 256):
    """Tokenize a raw text dataset into ``input_ids``/``attention_mask``/``labels``.

    Harden trainers consume model-ready tensors and drop every other column via
    ``_keep_model_columns``. A raw HF dataset (e.g. BeaverTails' ``prompt``/
    ``response``) must be tokenized first — otherwise all columns are stripped,
    ``remove_columns`` collapses the dataset to 0 rows, and training dies with
    ``num_samples=0``. If the dataset is already tokenized, it is returned as-is.
    """
    cols = set(dataset.column_names)
    if "input_ids" in cols:
        return dataset
    prompt_key = next((k for k in ("prompt", "instruction", "question", "query", "text")
                       if k in cols), None)
    resp_key = next((k for k in ("response", "output", "answer", "completion", "chosen")
                     if k in cols), None)
    if prompt_key is None:
        raise ValueError(
            f"Cannot tokenize training dataset with columns {sorted(cols)}: no "
            "recognizable prompt column. Provide a dataset with prompt/response-style "
            "columns, or one already tokenized to input_ids/attention_mask/labels."
        )
    from datasets import Dataset
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    rows = [(str(ex[prompt_key]), str(ex[resp_key]) if resp_key else "") for ex in dataset]
    tokenized = _tokenize_qa_rows(tok, rows, max_len)
    # Preserve a benign/harmful signal for safety-weighted trainers (STAR-DSS
    # reads a `kind` column; without it every row scores 0.0 and training
    # degenerates to a no-op). Non-safety trainers drop it via _keep_model_columns.
    safe_key = next((k for k in ("kind", "is_safe", "safe") if k in cols), None)
    if safe_key is not None:
        for row, ex in zip(tokenized, dataset):
            val = ex[safe_key]
            row["kind"] = val if safe_key == "kind" else ("benign" if bool(val) else "harmful")
    return Dataset.from_list(tokenized)


# ── Pillar handlers ───────────────────────────────────────────────────────────

# Harden trainers that exist but only work programmatically (DPO / adversarial /
# HF-Trainer interfaces that don't fit the uniform CLI train contract). Give a
# targeted pointer to the Python API instead of the generic "Unknown method".
_PROGRAMMATIC_ONLY_HARDEN = {
    "cst":         "CSTTrainer",
    "mart":        "MARTTrainer",
    "deeprefusal": "DeepRefusalTrainer",
    "antibody":    "AntibodyTrainer",
}


def _do_harden(args: argparse.Namespace) -> None:
    algo = args.algo.lower()
    if algo not in HARDEN_REGISTRY:
        if algo in _PROGRAMMATIC_ONLY_HARDEN:
            cls = _PROGRAMMATIC_ONLY_HARDEN[algo]
            print(
                f"{args.algo!r} is only available programmatically, not via the CLI: "
                f"it needs method-specific data/config that 'safetune train' can't supply. "
                f"Use it in Python, e.g.\n"
                f"    from safetune.runner.harden import {cls}\n"
                f"See the harden guide for the {cls} example."
            )
            sys.exit(1)
        print(f"Unknown harden method: {args.algo!r}. Run 'safetune list' to see all options.")
        sys.exit(1)

    from safetune.runner import harden

    model, tok = _load_model_and_tok(args.model)
    train_dataset = _ensure_tokenized(_load_train_dataset(args), tok)

    TrainerClass = getattr(harden, HARDEN_REGISTRY[algo])
    trainer = TrainerClass(model, tok, **_trainer_kwargs(args))
    out_path = trainer.train(train_dataset, out_dir=args.output)
    print(f"Saved to {out_path}")


def _do_eval(args: argparse.Namespace) -> None:
    from safetune.evaluate import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    benchmarks = args.dataset.split(",") if args.dataset else None

    results = evaluate(model, benchmarks=benchmarks, tokenizer=tokenizer)
    for name, metrics in results.items():
        print(f"{name}: {metrics}")


def _do_patch(args: argparse.Namespace) -> None:
    algo = args.algo.lower()
    if algo not in RECOVER_REGISTRY:
        print(f"Unknown recover method: {args.algo!r}. Run 'safetune list' to see all options.")
        sys.exit(1)

    from safetune.runner import recover
    from transformers import AutoModelForCausalLM

    model, tok = _load_model_and_tok(args.model)
    TrainerClass = getattr(recover, RECOVER_REGISTRY[algo])
    extra = {}
    if args.base:
        extra["base_model"] = AutoModelForCausalLM.from_pretrained(args.base)
    if args.aligned:
        extra["aligned_model"] = AutoModelForCausalLM.from_pretrained(args.aligned)
    if getattr(args, "alpha", None) is not None:
        extra["alpha"] = args.alpha
    extra.update(getattr(args, "method_kwargs", None) or {})
    trainer = TrainerClass(model, **extra)
    patched = trainer.apply()
    if args.output:
        out = args.output
        (patched if patched is not None else model).save_pretrained(out)
        tok.save_pretrained(out)
        print(f"Recover complete. Patched model saved to {out}")
    else:
        print("Recover complete (no --output given; patched model was not saved).")


def _do_unlearn(args: argparse.Namespace) -> None:
    algo = args.algo.lower()
    if algo not in UNLEARN_REGISTRY:
        print(f"Unknown unlearn method: {args.algo!r}. Run 'safetune list' to see all options.")
        sys.exit(1)

    from safetune.runner import unlearn
    model, tok = _load_model_and_tok(args.model)
    TrainerClass = getattr(unlearn, UNLEARN_REGISTRY[algo])
    # Unlearn trainers take `model` positionally and everything else keyword-only
    # (the tokenizer is derived from the model id), so do NOT pass `tok`
    # positionally — that would raise a TypeError before training starts.
    trainer = TrainerClass(model, model_id=args.model, **_trainer_kwargs(args))
    print(f"Unlearn trainer ready. Call trainer.unlearn(forget=..., retain=...) in Python.")


def _do_list(args: argparse.Namespace) -> None:
    """Print every available method, grouped by pillar."""
    def _section(title, registry):
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print(f"{'─' * 50}")
        for alias, cls_name in sorted(registry.items()):
            print(f"  {alias:<22}  →  {cls_name}")

    print("\nSafeTune — available methods")
    _section("HARDEN  (train-time)          safetune.runner.harden", HARDEN_REGISTRY)
    for alias, cls_name in sorted(_PROGRAMMATIC_ONLY_HARDEN.items()):
        print(f"  {alias:<22}  →  {cls_name}  (Python API only)")
    _section("RECOVER (weight-space)        safetune.runner.recover", RECOVER_REGISTRY)
    _section("UNLEARN (forget-set training) safetune.runner.unlearn", UNLEARN_REGISTRY)
    print(f"\n{'─' * 50}")
    print("  STEER and EVALUATE have no --algo flag; use the Python API.")
    print(f"{'─' * 50}\n")
    print("Examples:")
    print("  safetune train  --model Qwen/Qwen2.5-0.5B-Instruct --algo lisa")
    print("  safetune patch  --model ./drifted --algo resta --base ./base")
    print("  safetune unlearn --model ./model --algo rmu")
    print("  safetune eval   --model Qwen/Qwen2.5-0.5B-Instruct --dataset harmbench\n")


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    harden_keys  = ", ".join(sorted(HARDEN_REGISTRY))
    recover_keys = ", ".join(sorted(RECOVER_REGISTRY))
    unlearn_keys = ", ".join(sorted(UNLEARN_REGISTRY))

    parser = argparse.ArgumentParser(
        description="SafeTune CLI — safety alignment tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run 'safetune list' to see every available method.\n\n"
            "Harden methods:  " + harden_keys + "\n"
            "Recover methods: " + recover_keys + "\n"
            "Unlearn methods: " + unlearn_keys
        ),
    )
    parser.add_argument(
        "command",
        choices=["train", "eval", "patch", "unlearn", "list"],
        help="train (Harden) · eval (Evaluate) · patch (Recover) · unlearn (Unlearn) · list",
    )
    parser.add_argument("--config",      type=str, help="Path to a YAML config file (values are overridden by explicit CLI flags)")
    parser.add_argument("--algo",        default="safegrad", help="Method within the pillar")
    parser.add_argument("--model",       type=str, help="Model path or HF hub ID")
    parser.add_argument("--dataset",     type=str, help="Dataset or comma-separated benchmarks for eval")
    parser.add_argument("--output",      "--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--epochs",      type=int,   default=1,      help="Training epochs")
    parser.add_argument("--batch-size",  type=int,   default=1,      help="Batch size")
    parser.add_argument("--lr",          "--learning-rate", type=float, default=5e-5, help="Learning rate (train/unlearn)")
    parser.add_argument("--base",        type=str, help="Base model path (for recover)")
    parser.add_argument("--aligned",     type=str, help="Aligned model path (for recover)")
    parser.add_argument("--alpha",       type=float, default=None,
                        help="Interpolation strength for patch (recover) methods")
    parser.add_argument("--precision",   choices=["fp16", "bf16", "fp32"], default="bf16", help="Precision")
    parser.add_argument("--train-dataset", type=str, default="beavertails",
                        help="Training dataset: 'beavertails' (default) or any HF dataset id")
    parser.add_argument("--train-split",   type=str, default=None,
                        help="Split to load from --train-dataset "
                             "(default: 30k_train for beavertails, else 'train')")
    parser.add_argument("--eval-backend", type=str, default=None, choices=["vllm", "hf"],
                        help="Generation backend for evaluation "
                             "(default: auto — vLLM if installed, else hf)")
    parser.add_argument("--wandb",       action="store_true", help="Enable WandB")
    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_parser()

    # Pre-scan for --config before the full parse so we can inject its values
    # as parser defaults.  Explicit CLI flags still take precedence.
    pre, _ = parser.parse_known_args()
    method_kwargs: dict = {}
    if pre.config:
        from safetune.config import SafeTuneConfig
        cfg = SafeTuneConfig.from_yaml(pre.config)
        method_kwargs = dict(cfg.method_kwargs)
        cfg_ns = cfg.to_namespace()
        parser.set_defaults(**{
            k: v for k, v in vars(cfg_ns).items()
            if v is not None
        })

    args = parser.parse_args()
    # Carry method-specific kwargs through to the trainer constructors
    # (see _trainer_kwargs / _do_patch). Without this they are silently dropped.
    args.method_kwargs = method_kwargs
    return args


def main() -> None:
    _print_banner()
    args = parse_args()
    if args.command != "list" and not args.model:
        print("error: --model is required for this command.")
        sys.exit(1)
    # Apply the configured eval backend (if any) process-wide; None keeps auto
    # (vLLM when installed, else transformers).
    if getattr(args, "eval_backend", None):
        from safetune.runner.utils.eval_runner import set_eval_backend
        set_eval_backend(args.eval_backend)
    dispatch = {
        "train":   _do_harden,
        "eval":    _do_eval,
        "patch":   _do_patch,
        "unlearn": _do_unlearn,
        "list":    _do_list,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
