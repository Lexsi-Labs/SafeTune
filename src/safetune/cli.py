#!/usr/bin/env python3
"""SafeTune CLI: dispatch to each safety pillar."""

import argparse
import inspect
import os
import sys
from typing import Optional

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
        f"  {A}◆{R}  {B}{P}S A F E T U N E{R}  {D}v{__version__} · LSAL v1.2 · Python 3.12+{R}\n"
        f"     {D}Fine-tuning breaks safety. SafeTune fixes it.{R}\n"
        f"  {sep}\n"
        f"  {P}harden{R} · {P}recover{R} · {P}unlearn{R} · {P}steer{R}"
        f"  {D}· interpret · evaluate{R}\n"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model_and_tok(model_path: str):
    from transformers import AutoTokenizer
    from safetune._refusal_helpers import _load_pretrained_lm
    from safetune.config import resolve_dtype
    # Runtime dtype (fp32 on CPU): transformers 5 loads bf16 by default, and
    # bf16 training on CPU is very slow (110 s per step for Lisa on a 0.5B model).
    model = _load_pretrained_lm(model_path, dtype=resolve_dtype())
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
        # No --precision → None: the trainer derives it from the runtime dtype.
        bf16=(args.precision == "bf16") if args.precision else None,
        fp16=(args.precision == "fp16") if args.precision else None,
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

    A name from safetune.data.dataset_ids (e.g. ``beavertails``, split
    ``30k_train``) uses its table spec, including ``datasets:`` overrides; any
    other value is an HF id, local file, URL, or a dataset folder such as a
    CuratorKIT export with ``--train-config`` (``load_dataset(dir, config)``),
    split ``train``.
    """
    from safetune.data.dataset_ids import load, spec
    ds_name = getattr(args, "train_dataset", "beavertails") or "beavertails"
    split = getattr(args, "train_split", None)
    config = getattr(args, "train_config", None)
    if ds_name == "beavertails" and config is None:
        from safetune.data import load_beavertails
        return load_beavertails(split=split or spec("beavertails").get("split"))
    return load(ds_name, split=split, config=config)


def _load_safety_dataset(args, trainer, tok, max_len: Optional[int]):
    """``--safety-dataset`` / ``--safety-config`` / ``--safety-split`` as
    ``train(safety_dataset=)``, or None to keep the trainer's built-in safety set.
    A name from safetune.data.dataset_ids uses its table spec; any other value is
    an HF id, local file, URL or dataset folder (split ``train``)."""
    name = getattr(args, "safety_dataset", None)
    if not name:
        return None
    if "safety_dataset" not in inspect.signature(trainer.train).parameters:
        sys.exit(f"error: --algo {args.algo} takes no safety dataset; drop --safety-dataset.")
    from safetune.data.dataset_ids import load
    return _ensure_tokenized(load(name, split=getattr(args, "safety_split", None),
                                  config=getattr(args, "safety_config", None)), tok,
                             max_len=max_len)


def _ensure_tokenized(dataset, tok, *, max_len: Optional[int] = None):
    """``safetune.runner.utils.data_utils.tokenize_dataset`` (kept for callers)."""
    from safetune.runner.utils.data_utils import tokenize_dataset
    return tokenize_dataset(dataset, tok, max_len=max_len)


# ── Pillar handlers ───────────────────────────────────────────────────────────

def _do_harden(args: argparse.Namespace) -> None:
    algo = args.algo.lower()
    if algo not in HARDEN_REGISTRY:
        print(f"Unknown harden method: {args.algo!r}. Run 'safetune list' to see all options.")
        sys.exit(1)

    from safetune.runner import harden

    from safetune.config import get_config, resolve_device
    model, tok = _load_model_and_tok(args.model)
    model = model.to(resolve_device())  # custom-loop trainers train where the model is
    train_dataset = _ensure_tokenized(_load_train_dataset(args), tok,
                                      max_len=get_config().max_len)

    TrainerClass = getattr(harden, HARDEN_REGISTRY[algo])
    trainer = TrainerClass(model, tok, **_trainer_kwargs(args))
    safety_dataset = _load_safety_dataset(args, trainer, tok, get_config().max_len)
    extra = {} if safety_dataset is None else {"safety_dataset": safety_dataset}
    # The CLI loads the data itself, so it records the refs for lexsi_provenance.json.
    from safetune.provenance import input_entry
    trainer.dataset_inputs.append(input_entry(
        "dataset", args.train_dataset, getattr(args, "train_config", None)))
    if safety_dataset is not None:
        trainer.dataset_inputs.append(input_entry(
            "dataset", args.safety_dataset, getattr(args, "safety_config", None)))
    out_path = trainer.train(train_dataset, out_dir=args.output, **extra)
    print(f"Saved to {out_path}")


def _do_eval(args: argparse.Namespace) -> None:
    from safetune.evaluate import evaluate
    from safetune.evaluate.suite.benchmarks import check_benchmarks
    from safetune.runner.utils.model_utils import load_model
    from transformers import AutoTokenizer

    benchmarks = [b.strip() for b in args.dataset.split(",")] if args.dataset else None
    if benchmarks:
        try:
            check_benchmarks(benchmarks)  # before the model loads
        except ValueError as e:
            sys.exit(f"error: {e}")
    model = load_model(args.model)  # runtime device / dtype
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Run every benchmark, then exit non-zero if any of them failed.
    results = evaluate(model, benchmarks=benchmarks, tokenizer=tokenizer, strict=False)
    for name, metrics in results.items():
        print(f"{name}: FAILED: {metrics['error']}" if "error" in metrics else f"{name}: {metrics}")
    failed = [name for name, metrics in results.items() if "error" in metrics]
    if failed:
        sys.exit(f"safetune eval: {len(failed)} of {len(results)} benchmarks failed: "
                 f"{', '.join(failed)}")


def _do_patch(args: argparse.Namespace) -> None:
    algo = args.algo.lower()
    if algo not in RECOVER_REGISTRY:
        print(f"Unknown recover method: {args.algo!r}. Run 'safetune list' to see all options.")
        sys.exit(1)

    from safetune.runner import recover
    from safetune.config import resolve_dtype
    from safetune._refusal_helpers import _load_pretrained_lm

    model, tok = _load_model_and_tok(args.model)
    TrainerClass = getattr(recover, RECOVER_REGISTRY[algo])
    extra = {}
    if args.base:
        extra["base_model"] = _load_pretrained_lm(args.base, dtype=resolve_dtype())
    if args.aligned:
        extra["aligned_model"] = _load_pretrained_lm(args.aligned, dtype=resolve_dtype())
    if getattr(args, "alpha", None) is not None:
        extra["alpha"] = args.alpha
    extra.update(getattr(args, "method_kwargs", None) or {})
    trainer = TrainerClass(model, **extra)
    patched = trainer.apply()
    if args.output:
        from safetune.provenance import input_entry
        from safetune.runner.utils.model_utils import save_checkpoint
        out = os.path.normpath(args.output)
        # A standard HF folder (plus the processor for vision-language models).
        save_checkpoint(patched if patched is not None else model, tok,
                        os.path.basename(out), out_dir=os.path.dirname(out),
                        method=f"recover.{trainer.METHOD}",
                        inputs=[input_entry("model", p) for p in (args.base, args.aligned) if p])
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
    parser.add_argument("--precision",   choices=["fp16", "bf16", "fp32"], default=None,
                        help="Training precision (default: from the runtime dtype, bf16 where supported)")
    parser.add_argument("--train-dataset", type=str, default="beavertails",
                        help="Training dataset: 'beavertails' (default), an HF dataset id, "
                             "a local file, or a dataset folder such as a CuratorKIT export")
    parser.add_argument("--train-config", type=str, default=None,
                        help="Config of --train-dataset, e.g. sft_alpaca / sft_sharegpt / dpo "
                             "for a CuratorKIT export folder (load_dataset(dir, config))")
    parser.add_argument("--train-split",   type=str, default=None,
                        help="Split to load from --train-dataset "
                             "(default: 30k_train for beavertails, else 'train')")
    parser.add_argument("--safety-dataset", type=str, default=None,
                        help="Safety dataset for harden methods that take one (safegrad, "
                             "sap, ...): a name from safetune.data.dataset_ids or any HF id, "
                             "local file/directory or URL. Default: the method's built-in set")
    parser.add_argument("--safety-config", type=str, default=None,
                        help="Config of --safety-dataset (HF config or CuratorKIT export name)")
    parser.add_argument("--safety-split",  type=str, default=None,
                        help="Split to load from --safety-dataset (default: the table's "
                             "split, else 'train')")
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


def _apply_runtime(args: argparse.Namespace) -> None:
    """Apply the YAML ``runtime:`` / ``datasets:`` blocks process-wide, then
    --eval-backend (explicit flag wins; None keeps auto: vLLM if installed)."""
    from safetune.config import configure
    configure(**(getattr(args, "runtime", None) or {}))
    if getattr(args, "datasets", None):
        configure(datasets=args.datasets)
    if getattr(args, "eval_backend", None):
        configure(eval_backend=args.eval_backend)


def main() -> None:
    try:
        _main()
        sys.stdout.flush()
    except BrokenPipeError:
        # `safetune list | head` closed the pipe: stop quietly, like other CLIs.
        # When stdout is wrapped (colorama on Windows / FORCE_COLOR), the wrapper's
        # own __del__ can try to flush the already-broken stream during cleanup;
        # that second BrokenPipeError happens outside any reachable except block,
        # so Python reports it via sys.unraisablehook instead of propagating it --
        # silently drop that one specific, expected case while leaving every other
        # unraisable exception visible for debugging.
        def _ignore_broken_pipe_on_cleanup(unraisable):
            if not issubclass(unraisable.exc_type, BrokenPipeError):
                sys.__unraisablehook__(unraisable)

        sys.unraisablehook = _ignore_broken_pipe_on_cleanup
        # Point stdout at devnull so the interpreter's exit flush cannot raise again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        # A wrapped stdout (colormama's ansi2win32 on Windows) holds its own
        # buffer and flushes it at interpreter shutdown, after every except
        # clause has run -- and it writes to colormama's helper-process pipe,
        # not to fd 1, so the devnull redirect above cannot help. That flush
        # raises a second BrokenPipeError from __del__, which Python can only
        # report as "Exception ignored in: ..." with a full traceback on stderr.
        # os._exit() skips interpreter finalization entirely, so the wrapper's
        # exit flush never happens.
        os._exit(1)


def _main() -> None:
    _print_banner()
    args = parse_args()
    if args.command != "list" and not args.model:
        print("error: --model is required for this command.")
        sys.exit(1)
    _apply_runtime(args)
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
