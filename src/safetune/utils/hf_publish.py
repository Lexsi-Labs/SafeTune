"""
HuggingFace Hub publishing with SafeTune / Lexsi Labs model-card branding.

Same contract as AlignTune (`aligntune.utils.hf_publish`) and CuratorKIT
(`curatorkit.utils.hf_publish`): push weights, then `brand_hf_repo` uploads
packaged logos + a README. Image srcs point at *that* Hub repo — never a
personal CDN URL.

Public notebook cells call `push_to_hub` / `push_merged_to_hub` /
`push_quantized_to_hub` / `push_gguf_to_hub` (via :class:`HubPushMixin`) or
the standalone `push_*_path_to_hub` helpers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from huggingface_hub import HfApi

from .auth import get_hf_token

logger = logging.getLogger(__name__)

SAFETUNE_REPO_URL = "https://github.com/Lexsi-Labs/SafeTune"
LEXSI_URL = "https://lexsi.ai/"

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_BANNER_ASSET = _ASSETS_DIR / "safetune_banner.png"
_LOGO_ASSET = _ASSETS_DIR / "lexsi_logo.png"
_BANNER_REPO_NAME = "safetune_banner.png"
_LOGO_REPO_NAME = "lexsi_logo.png"

_VALID_KINDS = ("adapter", "model", "quant", "gguf", "tokenizer")


def is_hf_repo_id(value: str) -> bool:
    s = (value or "").strip()
    if not s or s in {"—", "-", "none", "None"}:
        return False
    if s.startswith(("/", ".", "~")) or "\\" in s:
        return False
    if len(s) >= 2 and s[1] == ":":
        return False
    parts = s.split("/")
    if not 1 <= len(parts) <= 2:
        return False
    return all(p and not p.startswith(".") for p in parts)


def resolve_hub_base_model(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if is_hf_repo_id(raw) and not os.path.exists(raw):
        return raw

    path = Path(raw)
    if not path.exists():
        return raw if is_hf_repo_id(raw) else ""

    candidates = []
    for name in ("adapter_config.json", "config.json"):
        f = path / name
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in (
            "base_model_name_or_path",
            "base_model",
            "_name_or_path",
            "name_or_path",
        ):
            cand = data.get(key)
            if cand:
                candidates.append(str(cand))

    seen = {os.path.abspath(raw)}
    for cand in candidates:
        if is_hf_repo_id(cand) and not os.path.exists(cand):
            return cand
        abs_cand = os.path.abspath(cand) if os.path.exists(cand) else ""
        if abs_cand and abs_cand not in seen:
            seen.add(abs_cand)
            nested = resolve_hub_base_model(cand)
            if nested:
                return nested
    return ""


def _resolve_token(token: Optional[str] = None) -> str:
    token = (
        token
        or get_hf_token()
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_LEXSI")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise ValueError(
            "No HuggingFace token found. Pass token=..., set HF_TOKEN / HF_LEXSI, "
            "or run `hf auth login`."
        )
    return token


def _hub_asset_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _upload_packaged_asset(
    api: HfApi, repo_id: str, token: str, local: Path, name: str
) -> Optional[str]:
    if not local.exists():
        logger.warning("Packaged branding asset missing: %s", local)
        return None
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=name,
        repo_id=repo_id,
        token=token,
    )
    return _hub_asset_url(repo_id, name)


def _branding_header(logo_url: Optional[str], banner_url: Optional[str]) -> str:
    cells = []
    if logo_url:
        cells.append(
            f"""      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{LEXSI_URL}"><img src="{logo_url}" alt="Lexsi Labs" style="height: 60px; border-radius: 12px;"/></a>
      </td>"""
        )
    if banner_url:
        cells.append(
            f"""      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{SAFETUNE_REPO_URL}"><img src="{banner_url}" alt="SafeTune" style="height: 60px;"/></a>
      </td>"""
        )
    if not cells:
        return ""
    inner = "\n".join(cells)
    return f"""<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; border-collapse: collapse;">
    <tr>
{inner}
    </tr>
  </table>
</div>
"""


def _usage_block(
    kind: str, repo_id: str, base_model: str, gguf_files: Optional[Iterable[str]]
) -> str:
    files = [str(x) for x in (gguf_files or [])]
    if kind == "adapter":
        return f"""```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

This repo is a LoRA adapter. Load it on top of `{base_model}` (PEFT does that from `adapter_config.json`)."""
    if kind == "quant":
        return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

This repo is a BitsAndBytes quantized checkpoint. Keep nf4 / bf4 / int8 in **separate** Hub repos — each save writes its own `config.json` `quantization_config` at the repo root."""
    if kind == "gguf":
        listed = "\n".join(f"- `{f}`" for f in files) if files else "- (GGUF files in this repo)"
        sample = files[0] if files else "model.gguf"
        return f"""Several GGUF files can live in **one** Hub repo (different filenames). That is the usual layout.

{listed}

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="{repo_id}", filename="{sample}")
```"""
    if kind == "tokenizer":
        return f"""```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```"""
    return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```"""


def brand_hf_repo(
    repo_id: str,
    kind: str = "model",
    base_model: str = "",
    algorithm: str = "",
    backend: str = "",
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
    gguf_files: Optional[Iterable[str]] = None,
) -> str:
    """Create the repo if needed, upload packaged logos, write README.md.

    Does not upload weights. Call after adapter / merged / quant / GGUF push.
    """
    kind = (kind or "model").lower()
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    token = _resolve_token(token)
    api = HfApi(token=token)
    built_on = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)

    logo_url = _upload_packaged_asset(api, repo_id, token, _LOGO_ASSET, _LOGO_REPO_NAME)
    banner_url = _upload_packaged_asset(
        api, repo_id, token, _BANNER_ASSET, _BANNER_REPO_NAME
    )
    header = _branding_header(logo_url, banner_url)

    model_name = repo_id.split("/")[-1]
    tag = (algorithm or "safetune").lower().replace(" ", "-").replace("(", "").replace(")", "")
    backend_tag = (backend or "safetune").lower()
    hub_base = resolve_hub_base_model(base_model) if base_model else ""
    usage = _usage_block(kind, repo_id, hub_base or base_model, gguf_files)
    if hub_base:
        base_row = (
            f"| **Built from** | [{hub_base}](https://huggingface.co/{hub_base}) |"
        )
        yaml_base = f"base_model: {hub_base}\n"
    elif base_model:
        base_row = f"| **Built from** | `{base_model}` |"
        yaml_base = ""
    else:
        base_row = "| **Built from** | — |"
        yaml_base = ""

    readme = f"""---
library_name: {"gguf" if kind == "gguf" else "transformers"}
{yaml_base}tags:
  - safetune
  - {tag}
  - {backend_tag}
  - {kind}
---

{header}
# {model_name}

Built using [SafeTune]({SAFETUNE_REPO_URL}) — a library of LLM-safety methods (harden / recover / unlearn / steer / interpret / evaluate).

| | |
|---|---|
{base_row}
| **Method** | {algorithm or "—"} |
| **Backend** | {backend or "safetune"} |
| **Artifact** | {kind} |
| **Published** | {built_on} |

{extra_notes}

## Usage

{usage}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
    finally:
        os.remove(readme_path)

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Branded %s", url)
    return url


def load_finetuned_model(
    output_dir: str,
    base_model: str,
    dtype: Any = "auto",
    device_map: Any = None,
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict = {"torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    is_lora = os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    if is_lora:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, **load_kwargs)
        model = model.merge_and_unload()
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
    else:
        model = AutoModelForCausalLM.from_pretrained(output_dir, **load_kwargs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    return model, tokenizer


def push_model_to_hf(
    model: Any,
    tokenizer: Any,
    repo_id: str,
    base_model: str,
    algorithm: str,
    backend: str = "safetune",
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    token = _resolve_token(token)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)
    model.push_to_hub(repo_id, token=token, private=private)
    tokenizer.push_to_hub(repo_id, token=token, private=private)
    return brand_hf_repo(
        repo_id,
        kind="model",
        base_model=base_model,
        algorithm=algorithm,
        backend=backend,
        private=private,
        token=token,
        extra_notes=extra_notes,
    )


def _brand_src(repo_id, kind, src, private, token, base_model="", **kw):
    return brand_hf_repo(
        repo_id,
        kind=kind,
        private=private,
        token=token,
        base_model=base_model
        or resolve_hub_base_model(src)
        or (src if is_hf_repo_id(src) else ""),
        **kw,
    )


def _upload_folder(folder, repo_id, private, token):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Local folder not found: {folder}")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        token=token,
        ignore_patterns=[".git*", "__pycache__", "*.pyc"],
    )


def push_folder_to_hub(folder, repo_id, private=False, token=None, **kw):
    token = _resolve_token(token)
    _upload_folder(folder, repo_id, private, token)
    kind = "adapter" if os.path.exists(os.path.join(folder, "adapter_config.json")) else "model"
    return _brand_src(repo_id, kind, folder, private, token, **kw)


def push_tokenizer_path_to_hub(path, repo_id, private=False, token=None, **kw):
    token = _resolve_token(token)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(path).push_to_hub(repo_id, token=token, private=private)
    return _brand_src(repo_id, "tokenizer", path, private, token, **kw)


def push_model_path_to_hub(path, repo_id, private=False, token=None, tokenizer_path=None, **kw):
    token = _resolve_token(token)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.isdir(path):
        _upload_folder(path, repo_id, private, token)
        if tokenizer_path:
            AutoTokenizer.from_pretrained(tokenizer_path).push_to_hub(
                repo_id, token=token, private=private
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")
        tok = AutoTokenizer.from_pretrained(tokenizer_path or path)
        try:
            model.push_to_hub(repo_id, token=token, private=private)
            tok.push_to_hub(repo_id, token=token, private=private)
        finally:
            del model
    return _brand_src(repo_id, "model", path, private, token, **kw)


def push_quantized_path_to_hub(
    path,
    repo_id,
    quantization="nf4",
    private=False,
    token=None,
    tokenizer_path=None,
    **kw,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = _resolve_token(token)
    presets = {
        "nf4": dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "fp4": dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "bf4": dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "int8": dict(load_in_8bit=True),
    }
    key = quantization.lower()
    if key not in presets:
        raise ValueError(f"quantization must be one of {sorted(presets)}, got {quantization!r}")
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=BitsAndBytesConfig(**presets[key]), device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_path or path)
    try:
        model.push_to_hub(repo_id, token=token, private=private)
        tok.push_to_hub(repo_id, token=token, private=private)
    finally:
        del model
    return _brand_src(
        repo_id,
        "quant",
        path,
        private,
        token,
        extra_notes=f"BitsAndBytes `{key}` quantization.",
        **kw,
    )


def push_gguf_path_to_hub(
    path, repo_id, quantization="Q5_K_M", private=False, token=None, **kw
):
    """Upload existing GGUF files, or convert with llama.cpp if CONVERT_HF_TO_GGUF is set."""
    token = _resolve_token(token)
    quant = str(quantization).upper()
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)

    src = Path(path)
    existing = []
    if src.is_file() and src.suffix.lower() == ".gguf":
        existing = [src]
    elif src.is_dir():
        existing = sorted(src.glob("*.gguf"))

    uploaded = []
    if existing:
        for f in existing:
            name = f.name
            api.upload_file(
                path_or_fileobj=str(f), path_in_repo=name, repo_id=repo_id, token=token
            )
            uploaded.append(name)
    else:
        convert = os.environ.get("CONVERT_HF_TO_GGUF")
        if not convert:
            raise RuntimeError(
                "No .gguf files in {path!r}. Set CONVERT_HF_TO_GGUF to llama.cpp "
                "convert_hf_to_gguf.py, or export GGUF first."
            )
        import subprocess

        out_name = f"model-{quant.lower()}.gguf"
        out_path = Path(f"./gguf_{quant}") / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["python", convert, str(src), "--outfile", str(out_path), "--outtype", quant.lower()]
        )
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=out_name,
            repo_id=repo_id,
            token=token,
        )
        uploaded = [out_name]

    return _brand_src(
        repo_id, "gguf", path, private, token, gguf_files=uploaded, **kw
    )


class HubPushMixin:
    """push_to_hub / merged / quantized / GGUF — same names as AlignTune TrainerBase."""

    _hub_algorithm = "safetune"
    _hub_backend = "safetune"
    _hub_merge_dir = "./out_merged"

    def _hub_tokenizer(self):
        return (
            getattr(self, "tok", None)
            or getattr(self, "tokenizer", None)
            or getattr(self, "processing_class", None)
        )

    def _hub_meta(self):
        name = str(getattr(self, "model_id", "") or "")
        if not name:
            cfg = getattr(getattr(self, "model", None), "config", None)
            name = str(getattr(cfg, "_name_or_path", "") or "")
        algo = getattr(self, "METHOD", None) or getattr(self, "_hub_algorithm", "safetune")
        return {
            "base_model": resolve_hub_base_model(name) or name,
            "algorithm": str(algo),
            "backend": getattr(self, "_hub_backend", "safetune"),
        }

    def _hub_ckpt(self) -> str:
        cached = getattr(self, "_last_ckpt", None) or getattr(self, "_merged_path", None)
        if cached:
            return cached
        return getattr(self, "_hub_merge_dir", "./out_merged")

    def _merged_checkpoint(self) -> str:
        cached = getattr(self, "_merged_path", None) or getattr(self, "_last_ckpt", None)
        if cached and os.path.isdir(cached):
            return cached
        path = getattr(self, "_hub_merge_dir", "./out_merged")
        Path(path).mkdir(parents=True, exist_ok=True)
        tok = self._hub_tokenizer()
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError("No model on trainer to merge/save.")
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged.save_pretrained(path)
            model = merged
        else:
            model.save_pretrained(path)
        if tok is not None and hasattr(tok, "save_pretrained"):
            tok.save_pretrained(path)
        self._merged_path = path
        return path

    def push_to_hub(
        self, repo_id, private=False, token=None, commit_message="Upload model", **kwargs
    ):
        ckpt = self._hub_ckpt()
        if os.path.isdir(ckpt):
            url = push_folder_to_hub(
                ckpt, repo_id, private=private, token=token, **self._hub_meta()
            )
        else:
            tok = self._hub_tokenizer()
            self.model.push_to_hub(
                repo_id, private=private, commit_message=commit_message, token=token
            )
            if tok is not None and hasattr(tok, "push_to_hub"):
                tok.push_to_hub(
                    repo_id, private=private, commit_message=commit_message, token=token
                )
            kind = "adapter" if hasattr(self.model, "peft_config") else "model"
            try:
                brand_hf_repo(
                    repo_id, kind=kind, private=private, token=token, **self._hub_meta()
                )
            except Exception as e:
                logger.warning("Model card branding skipped for %s: %s", repo_id, e)
            url = f"https://huggingface.co/{repo_id}"
        print(url)
        return url

    def push_merged_to_hub(self, repo_id, private=False, token=None, max_shard_size="2GB"):
        url = push_model_path_to_hub(
            self._merged_checkpoint(),
            repo_id,
            private=private,
            token=token,
            **self._hub_meta(),
        )
        print(url)
        return url

    def push_quantized_to_hub(self, repo_id, quantization="nf4", private=False, token=None):
        url = push_quantized_path_to_hub(
            self._merged_checkpoint(),
            repo_id,
            quantization,
            private=private,
            token=token,
            **self._hub_meta(),
        )
        print(url)
        return url

    def push_gguf_to_hub(self, repo_id, quantizations, private=False, token=None):
        if isinstance(quantizations, str):
            quantizations = [quantizations]
        url = ""
        for quant in quantizations:
            url = push_gguf_path_to_hub(
                self._merged_checkpoint(),
                repo_id,
                quant,
                private=private,
                token=token,
                **self._hub_meta(),
            )
        print(url)
        return url


class FolderPublisher(HubPushMixin):
    """Same ``push_to_hub`` / merged / quantized / GGUF methods as trainers.

    Use in notebooks that write a checkpoint folder but do not subclass a
    SafeTune trainer (interpret, CLI, red-team, TamperBench).
    """

    def __init__(
        self,
        ckpt,
        *,
        model_id: str = "",
        algorithm: str = "safetune",
        tokenizer=None,
        model=None,
    ):
        self._last_ckpt = ckpt
        self.model_id = model_id
        self.METHOD = algorithm
        self.tok = tokenizer
        self.model = model

