"""
Training orchestration logic for SafeTune.
Provides functional APIs for SFT, DPO, PPO, and GRPO training.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# `safetune.core.backend_factory` is not implemented. Guard the import so this
# module (and the `safetune` CLI that imports it) loads cleanly; the run_*
# functions then fail loudly with guidance instead of crashing at import time.
try:
    from safetune.core.backend_factory import create_sft_trainer, create_rl_trainer
except ImportError:  # pragma: no cover
    def _backend_unavailable(*_args, **_kwargs):
        raise NotImplementedError(
            "SafeTune's generic training-orchestration backend "
            "(safetune.core.backend_factory) is not implemented. Use a pillar "
            "entry point directly instead — e.g. a HARDEN trainer "
            "(safetune.harden.SafeGradTrainer, ...) or a standard "
            "transformers/TRL Trainer."
        )
    create_sft_trainer = create_rl_trainer = _backend_unavailable

def build_base_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and build base configuration."""
    return {
        'experiment_name': config_dict.get('experiment_name', 'safetune_experiment'),
        'output_dir': config_dict.get('output_dir', './results'),
        'seed': config_dict.get('seed', 3407),
        'logging': {
            'use_wandb': config_dict.get('wandb', False),
            'use_tensorboard': config_dict.get('tensorboard', False),
            'log_level': config_dict.get('log_level', 'INFO'),
        }
    }

def run_sft(
    model_name: str,
    dataset_name: str,
    output_dir: str = "./output",
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_seq_length: int = 512,
    **kwargs
):
    """Run Supervised Fine-Tuning."""
    print(f"\n🚀 SafeTune: Starting SFT on {model_name}")
    trainer = create_sft_trainer(
        model_name=model_name,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        **kwargs
    )
    return trainer.train()

def run_dpo(
    model_name: str,
    dataset_name: str,
    output_dir: str = "./output",
    beta: float = 0.1,
    **kwargs
):
    """Run Direct Preference Optimization."""
    print(f"\n🚀 SafeTune: Starting DPO on {model_name}")
    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=dataset_name,
        algorithm='dpo',
        output_dir=output_dir,
        beta=beta,
        **kwargs
    )
    return trainer.train()

def run_ppo(
    model_name: str,
    dataset_name: str,
    output_dir: str = "./output",
    **kwargs
):
    """Run Proximal Policy Optimization."""
    print(f"\n🚀 SafeTune: Starting PPO on {model_name}")
    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=dataset_name,
        algorithm='ppo',
        output_dir=output_dir,
        **kwargs
    )
    return trainer.train()

def run_grpo(
    model_name: str,
    dataset_name: str,
    output_dir: str = "./output",
    **kwargs
):
    """Run Group Relative Policy Optimization."""
    print(f"\n🚀 SafeTune: Starting GRPO on {model_name}")
    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=dataset_name,
        algorithm='grpo',
        output_dir=output_dir,
        **kwargs
    )
    return trainer.train()
