"""Push SafeTune outputs to the Hugging Face Hub."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from safetune.provenance import FILENAME


def push_to_hub(path: str, repo_id: str, *, repo_type: str = "model",
                private: Optional[bool] = None, token: Optional[str] = None,
                path_in_repo: Optional[str] = None,
                commit_message: str = "Upload with SafeTune"):
    """Upload a SafeTune output to ``repo_id`` (created if missing).

    ``path`` is a checkpoint folder (model, tokenizer, processor and
    ``lexsi_provenance.json``, as ``save_checkpoint`` writes it) or a single file
    such as a results JSON, which is uploaded together with the
    ``lexsi_provenance.json`` next to it. ``token`` defaults to the logged-in
    ``huggingface_hub`` token. Returns the ``upload_folder`` commit info.
    """
    from huggingface_hub import HfApi

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"push_to_hub: {path} does not exist")
    folder = p if p.is_dir() else p.parent
    if not (folder / FILENAME).is_file():
        warnings.warn(f"push_to_hub: {folder} has no {FILENAME}; uploading without provenance.")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id, repo_type=repo_type, folder_path=str(folder),
        path_in_repo=path_in_repo, commit_message=commit_message,
        allow_patterns=None if p.is_dir() else [p.name, FILENAME])
