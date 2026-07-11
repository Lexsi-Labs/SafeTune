"""
HuggingFace Hub Integration for Safety Configs.

Push/pull safety configurations and steering vectors to/from HuggingFace Hub,
enabling sharing and versioning of safety setups across teams.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HFSafetyCard:
    """Safety card metadata for a model on HuggingFace Hub."""

    model_name: str = ""
    safety_score: float = 0.0
    benchmarks_passed: List[str] = None
    safety_modules_used: List[str] = None
    steering_vectors_available: bool = False
    evaluation_date: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.benchmarks_passed is None:
            self.benchmarks_passed = []
        if self.safety_modules_used is None:
            self.safety_modules_used = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "safety_score": self.safety_score,
            "benchmarks_passed": self.benchmarks_passed,
            "safety_modules_used": self.safety_modules_used,
            "steering_vectors_available": self.steering_vectors_available,
            "evaluation_date": self.evaluation_date,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Safety Card",
            "",
            f"**Model**: {self.model_name}",
            f"**Safety Score**: {self.safety_score:.2f}",
            f"**Evaluation Date**: {self.evaluation_date}",
            "",
            "## Benchmarks Passed",
        ]
        for b in self.benchmarks_passed:
            lines.append(f"- {b}")
        lines.extend(["", "## Safety Modules Used"])
        for m in self.safety_modules_used:
            lines.append(f"- {m}")
        if self.steering_vectors_available:
            lines.extend(["", "## Steering Vectors", "Steering vectors are available for inference-time safety."])
        if self.notes:
            lines.extend(["", "## Notes", self.notes])
        return "\n".join(lines)


class HFSafetyIntegration:
    """Push/pull safety configs and vectors to HuggingFace Hub."""

    SAFETY_CONFIG_FILENAME = "safety_config.json"
    SAFETY_CARD_FILENAME = "SAFETY_CARD.md"
    STEERING_VECTORS_FILENAME = "steering_vectors.pt"

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token or os.environ.get("HF_TOKEN")

    def _get_api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "HuggingFace Hub integration requires huggingface_hub. "
                "Install with: pip install huggingface-hub"
            )
        return HfApi(token=self.token)

    def push_safety_config(
        self,
        repo_id: str,
        config_dict: Dict[str, Any],
        safety_card: Optional[HFSafetyCard] = None,
        commit_message: str = "Update safety configuration",
    ) -> str:
        """Push a safety config to a HuggingFace repo.

        Args:
            repo_id: HuggingFace repo (e.g. "user/model-name").
            config_dict: the UnifiedSafetyConfig as dict.
            safety_card: optional safety card metadata.
            commit_message: git commit message.

        Returns:
            URL of the uploaded file.
        """
        api = self._get_api()

        # save config to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f, indent=2)
            config_path = f.name

        try:
            url = api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo=self.SAFETY_CONFIG_FILENAME,
                repo_id=repo_id,
                commit_message=commit_message,
            )
            logger.info("Pushed safety config to %s", repo_id)
        finally:
            os.unlink(config_path)

        # push safety card if provided
        if safety_card:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(safety_card.to_markdown())
                card_path = f.name
            try:
                api.upload_file(
                    path_or_fileobj=card_path,
                    path_in_repo=self.SAFETY_CARD_FILENAME,
                    repo_id=repo_id,
                    commit_message="Update safety card",
                )
            finally:
                os.unlink(card_path)

        return url

    def pull_safety_config(self, repo_id: str) -> Dict[str, Any]:
        """Pull a safety config from a HuggingFace repo."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required.")

        path = hf_hub_download(
            repo_id=repo_id,
            filename=self.SAFETY_CONFIG_FILENAME,
            token=self.token,
        )
        with open(path, "r") as f:
            config = json.load(f)
        logger.info("Pulled safety config from %s", repo_id)
        return config

    def push_steering_vectors(
        self,
        repo_id: str,
        vectors_path: str,
        commit_message: str = "Upload steering vectors",
    ) -> str:
        """Push precomputed steering vectors to HuggingFace repo."""
        api = self._get_api()
        url = api.upload_file(
            path_or_fileobj=vectors_path,
            path_in_repo=self.STEERING_VECTORS_FILENAME,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info("Pushed steering vectors to %s", repo_id)
        return url

    def pull_steering_vectors(self, repo_id: str, save_path: str) -> str:
        """Pull steering vectors from HuggingFace repo."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required.")

        path = hf_hub_download(
            repo_id=repo_id,
            filename=self.STEERING_VECTORS_FILENAME,
            token=self.token,
        )
        # copy to save_path
        import shutil
        shutil.copy(path, save_path)
        logger.info("Pulled steering vectors from %s to %s", repo_id, save_path)
        return save_path

    def create_safety_card(
        self,
        model_name: str,
        safety_score: float,
        benchmarks: List[str],
        modules: List[str],
        has_vectors: bool = False,
    ) -> HFSafetyCard:
        """Create a safety card from evaluation results."""
        from datetime import datetime
        return HFSafetyCard(
            model_name=model_name,
            safety_score=safety_score,
            benchmarks_passed=benchmarks,
            safety_modules_used=modules,
            steering_vectors_available=has_vectors,
            evaluation_date=datetime.now().strftime("%Y-%m-%d"),
        )
