"""
Enhanced error handling utilities for SafeTune.


This module provides user-friendly error messages, structured exceptions,
and error recovery suggestions.
"""

import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class SafeTuneError(Exception):
    """Base exception class for SafeTune errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions or []
        super().__init__(self.message)

    def __str__(self):
        return self.format_error()

    def format_error(self) -> str:
        """Format error message with suggestions."""
        lines = [f"❌ {self.message}"]

        if self.error_code:
            lines.append(f"   Error Code: {self.error_code}")

        if self.suggestions:
            lines.append("   💡 Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"      • {suggestion}")

        return "\n".join(lines)


class ConfigurationError(SafeTuneError):
    """Configuration-related errors."""

    def __init__(self, message: str, field: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.field = field
        error_code = "CONFIG_ERROR"
        if not suggestions:
            suggestions = self._generate_config_suggestions(message, field)
        super().__init__(message, error_code, suggestions)

    def _generate_config_suggestions(self, message: str, field: Optional[str]) -> List[str]:
        """Generate configuration-specific suggestions."""
        suggestions = []

        if "optimizer" in message.lower():
            suggestions.extend([
                "Check available optimizers with 'SafeTune recipes list'",
                "Use 'adamw_torch' for general training",
                "Use 'adamw_8bit' for memory-efficient training"
            ])

        if "scheduler" in message.lower():
            suggestions.extend([
                "Check available schedulers with 'SafeTune recipes list'",
                "Use 'cosine' for most training scenarios",
                "Use 'linear' for fine-tuning tasks"
            ])

        if "model" in message.lower() and "not found" in message.lower():
            suggestions.extend([
                "Ensure model name is correct (e.g., 'meta-llama/Meta-Llama-3-8B')",
                "Run 'huggingface-cli login' if using private models",
                "Check model availability at https://huggingface.co/models"
            ])

        if "dataset" in message.lower() and "not found" in message.lower():
            suggestions.extend([
                "Check dataset name is correct (e.g., 'tatsu-lab/alpaca')",
                "Verify dataset exists at https://huggingface.co/datasets"
            ])

        if "memory" in message.lower():
            suggestions.extend([
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use 4-bit quantization",
                "Reduce model max length"
            ])

        if field:
            suggestions.append(f"Check the '{field}' field in your configuration")

        return suggestions


class TrainingError(SafeTuneError):
    """Training-related errors."""

    def __init__(self, message: str, stage: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.stage = stage
        error_code = "TRAINING_ERROR"
        if not suggestions:
            suggestions = self._generate_training_suggestions(message, stage)
        super().__init__(message, error_code, suggestions)

    def _generate_training_suggestions(self, message: str, stage: Optional[str]) -> List[str]:
        """Generate training-specific suggestions."""
        suggestions = []

        if "cuda" in message.lower() or "gpu" in message.lower():
            suggestions.extend([
                "Check GPU memory availability with 'nvidia-smi'",
                "Reduce batch size or enable gradient accumulation",
                "Enable gradient checkpointing",
                "Use mixed precision (bf16/fp16)"
            ])

        if "out of memory" in message.lower():
            suggestions.extend([
                "Reduce per_device_batch_size",
                "Increase gradient_accumulation_steps",
                "Enable gradient_checkpointing",
                "Use 4-bit quantization",
                "Reduce max_seq_length"
            ])

        if "nan" in message.lower() or "inf" in message.lower():
            suggestions.extend([
                "Reduce learning rate",
                "Enable gradient clipping",
                "Check input data quality",
                "Use more stable optimizer (AdamW instead of Lion)"
            ])

        if "loss" in message.lower() and "not decreasing" in message.lower():
            suggestions.extend([
                "Increase learning rate",
                "Check data quality and preprocessing",
                "Try different optimizer",
                "Adjust batch size"
            ])

        if stage:
            suggestions.append(f"Error occurred during {stage} stage")

        return suggestions


class EnvironmentError(SafeTuneError):
    """Environment and dependency errors."""

    def __init__(self, message: str, dependency: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.dependency = dependency
        error_code = "ENV_ERROR"
        if not suggestions:
            suggestions = self._generate_env_suggestions(message, dependency)
        super().__init__(message, error_code, suggestions)

    def _generate_env_suggestions(self, message: str, dependency: Optional[str]) -> List[str]:
        """Generate environment-specific suggestions."""
        suggestions = []

        if "torch" in message.lower() or "cuda" in message.lower():
            suggestions.extend([
                "Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "Check CUDA version compatibility",
                "Update GPU drivers"
            ])

        if "transformers" in message.lower():
            suggestions.extend([
                "Install transformers: pip install transformers",
                "Update to latest version: pip install --upgrade transformers"
            ])

        if "trl" in message.lower():
            suggestions.extend([
                "Install TRL: pip install trl",
                "Update to latest version: pip install --upgrade trl"
            ])

        if "bitsandbytes" in message.lower():
            suggestions.extend([
                "Install bitsandbytes: pip install bitsandbytes",
                "For CUDA issues, try: pip install bitsandbytes --index-url https://download.pytorch.org/whl/cu118"
            ])

        if "unsloth" in message.lower():
            suggestions.extend([
                "Install unsloth: pip install unsloth",
                "Check CUDA compatibility for your GPU"
            ])

        if dependency:
            suggestions.append(f"Install missing dependency: pip install {dependency}")

        return suggestions


class GatedResourceError(SafeTuneError):
    """A Hugging Face Hub model or dataset this run needs is gated, private, or
    otherwise inaccessible with the current credentials.

    Carries ``repo_id`` and ``kind`` as attributes (not just message text) so a
    caller -- a training service, a UI wizard -- can catch this one error code
    and react (e.g. tell the user which access to request) instead of parsing
    ``huggingface_hub`` error text or crashing on a raw ``OSError``.
    """

    def __init__(self, repo_id: str, kind: str = "model", cause: Optional[Exception] = None):
        self.repo_id = repo_id
        self.kind = kind
        self.cause = cause
        gated = _is_gated_repo_error(cause) if cause is not None else True
        if gated:
            message = (f"{kind} '{repo_id}' is gated on Hugging Face Hub; this "
                       f"token/account has not been granted access.")
            suggestions = [
                f"Request access at https://huggingface.co/{repo_id}",
                "Once approved, run 'huggingface-cli login' with a token for that "
                "account, or pass token= / set the HF_TOKEN environment variable",
            ]
        else:
            message = (f"{kind} '{repo_id}' could not be found or accessed on "
                      f"Hugging Face Hub with the current credentials.")
            suggestions = [
                f"Check the {kind} id is correct: https://huggingface.co/{repo_id}",
                "If it's private, run 'huggingface-cli login' with a token that has access",
            ]
        super().__init__(message, error_code="GATED_RESOURCE", suggestions=suggestions)


def _find_in_cause_chain(exc: Exception, cls) -> Optional[Exception]:
    """Walk ``__cause__``/``__context__`` for an instance of ``cls``. vLLM (and
    sometimes transformers) wraps the original ``huggingface_hub`` error in its
    own exception type rather than letting it propagate directly, so a plain
    ``isinstance``/``except`` on the outer exception misses it."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, cls):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def _is_gated_repo_error(exc: Exception) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:  # older huggingface_hub layout
        from huggingface_hub.utils import GatedRepoError
    return _find_in_cause_chain(exc, GatedRepoError) is not None


@contextmanager
def hf_access_errors(repo_id: str, kind: str = "model"):
    """Wrap a Hugging Face Hub load of ``repo_id``: translate a gated or
    otherwise inaccessible repo failure into :class:`GatedResourceError`
    (``error_code`` ``"GATED_RESOURCE"``) instead of letting the raw
    ``huggingface_hub``/``OSError`` (or a vLLM/transformers wrapper around it)
    propagate, so callers can catch one stable error instead of parsing Hub
    error text.

    Usage::

        with hf_access_errors("meta-llama/Llama-3.1-8B-Instruct", kind="judge model"):
            AutoTokenizer.from_pretrained(model_id)
    """
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # older huggingface_hub layout
        from huggingface_hub.utils import RepositoryNotFoundError
    try:
        yield
    except Exception as e:
        found = _find_in_cause_chain(e, RepositoryNotFoundError)
        if found is None:
            raise
        raise GatedResourceError(repo_id, kind, cause=found) from e


class ValidationError(SafeTuneError):
    """Validation-related errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, suggestions: Optional[List[str]] = None):
        self.field = field
        self.value = value
        error_code = "VALIDATION_ERROR"
        if not suggestions:
            suggestions = self._generate_validation_suggestions(message, field, value)
        super().__init__(message, error_code, suggestions)

    def _generate_validation_suggestions(self, message: str, field: Optional[str], value: Optional[Any]) -> List[str]:
        """Generate validation-specific suggestions."""
        suggestions = []

        if "positive" in message.lower() and field:
            suggestions.append(f"Set {field} to a positive value (current: {value})")

        if "required" in message.lower() and field:
            suggestions.append(f"Provide a value for the required field '{field}'")

        if "invalid" in message.lower():
            suggestions.append("Check the allowed values in the documentation")

        if field:
            suggestions.append(f"Fix the '{field}' field in your configuration")

        return suggestions


def handle_error(error: Exception, verbose: bool = False) -> str:
    """
    Handle and format errors for user display.

    Args:
        error: The exception that occurred
        verbose: Whether to show verbose error information

    Returns:
        Formatted error message
    """
    if isinstance(error, SafeTuneError):
        return str(error)
    else:
        # Handle standard Python exceptions
        error_msg = f"❌ {str(error)}"

        # Add suggestions for common errors
        suggestions = []

        if "No module named" in str(error):
            module = str(error).split("'")[1] if "'" in str(error) else "unknown"
            suggestions.extend([
                f"Install missing module: pip install {module}",
                "Check your Python environment",
                "Run 'pip list' to see installed packages"
            ])

        if "CUDA out of memory" in str(error):
            suggestions.extend([
                "Reduce batch size",
                "Enable gradient accumulation",
                "Use gradient checkpointing",
                "Switch to CPU training for testing"
            ])

        if "Connection" in str(error) or "timeout" in str(error).lower():
            suggestions.extend([
                "Check internet connection",
                "Try again later",
                "Use a local model/dataset if available"
            ])

        if suggestions:
            error_msg += "\n   💡 Suggestions:"
            for suggestion in suggestions:
                error_msg += f"\n      • {suggestion}"

        if verbose:
            import traceback
            error_msg += f"\n\nDetailed traceback:\n{traceback.format_exc()}"

        return error_msg


def create_progress_display():
    """
    Create an enhanced progress display with health monitoring.

    Returns:
        Configured progress display
    """
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        MofNCompleteColumn
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        refresh_per_second=2,
    )


class HealthMonitor:
    """Monitor training health and provide early warnings."""

    def __init__(self):
        self.warnings = []
        self.critical_issues = []
        self.last_loss = None
        self.loss_history = []
        self.consecutive_bad_steps = 0

    def update_metrics(self, step: int, loss: float, lr: float) -> List[str]:
        """
        Update metrics and check for issues.

        Args:
            step: Current training step
            loss: Current loss value
            lr: Current learning rate

        Returns:
            List of warnings/issues detected
        """
        alerts = []

        # Track loss history
        self.loss_history.append(loss)
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)

        # Check for NaN/Inf loss
        if not (loss >= 0 and loss < float('inf')):
            alerts.append("Loss became NaN or infinite")
            self.critical_issues.append(f"Step {step}: Invalid loss value {loss}")

        # Check for loss spikes
        if len(self.loss_history) >= 3:
            recent_avg = sum(self.loss_history[-3:]) / 3
            if loss > recent_avg * 3:  # 3x spike
                alerts.append(f"Loss spike detected (current: {loss:.4f}, recent avg: {recent_avg:.4f})")

        # Check for no improvement
        if len(self.loss_history) >= 5:
            if all(l >= loss for l in self.loss_history[-4:]):  # Loss not decreasing
                self.consecutive_bad_steps += 1
                if self.consecutive_bad_steps >= 5:
                    alerts.append("Loss has not decreased for 5+ consecutive steps")
            else:
                self.consecutive_bad_steps = 0

        # Check learning rate
        if lr <= 1e-8:
            alerts.append("Learning rate is very low, training may be ineffective")

        # Store alerts as warnings
        for alert in alerts:
            self.warnings.append(f"Step {step}: {alert}")

        return alerts

    def get_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        return {
            "total_warnings": len(self.warnings),
            "critical_issues": len(self.critical_issues),
            "recent_warnings": self.warnings[-5:] if self.warnings else [],
            "loss_trend": "improving" if self.consecutive_bad_steps == 0 else "stalled"
        }


# Convenience functions for common error types
def config_error(message: str, field: Optional[str] = None) -> ConfigurationError:
    """Create a configuration error."""
    return ConfigurationError(message, field)


def training_error(message: str, stage: Optional[str] = None) -> TrainingError:
    """Create a training error."""
    return TrainingError(message, stage)


def env_error(message: str, dependency: Optional[str] = None) -> EnvironmentError:
    """Create an environment error."""
    return EnvironmentError(message, dependency)


def validation_error(message: str, field: Optional[str] = None, value: Optional[Any] = None) -> ValidationError:
    """Create a validation error."""
    return ValidationError(message, field, value)


def format_validation_errors(errors: List[str], warnings: List[str]) -> str:
    """Format validation errors and warnings for display."""

    if not errors and not warnings:
        return "[green]✅ Configuration is valid![/green]"

    output = Text()

    if errors:
        output.append("\n❌ Errors:\n", style="red bold")
        for error in errors:
            output.append(f"  • {error}\n", style="red")

    if warnings:
        output.append("\n⚠️  Warnings:\n", style="yellow bold")
        for warning in warnings:
            output.append(f"  • {warning}\n", style="yellow")

    if errors:
        output.append("\n[red]Please fix the errors before proceeding.[/red]")
    elif warnings:
        output.append("\n[yellow]Consider addressing the warnings for better performance.[/yellow]")

    return str(output)
