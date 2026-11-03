"""Audit H21: the runner's RepNoise MMD weight must default to the paper's 0.001.

Rosati et al. 2024 (arXiv:2405.14577, appendix) and domenicrosati/representation-noising @03689ff
`loss.py:74` use beta = 0.001 for the MMD noise term; RepNoiseConfig.repnoise_beta3 is 0.001. The runner
defaulted noise_beta to 0.1 (100x) with no reason in the docstring or the user guide, whose own example
passes noise_beta=0.001.
"""
import inspect

import safetune.harden as H
from safetune.runner.harden import RepNoiseTrainer


def test_runner_noise_beta_default_is_the_papers_and_the_trainers():
    runner = inspect.signature(RepNoiseTrainer.__init__).parameters["noise_beta"].default
    assert runner == 0.001
    assert runner == H.RepNoiseConfig(output_dir="/tmp/h21_cfg").repnoise_beta3
