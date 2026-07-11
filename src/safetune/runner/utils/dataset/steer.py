from safetune.runner.utils.data_utils import refusal_prompt_pairs_large


def load_steer_dataset(n: int = 256) -> tuple[list[str], list[str]]:
    """Harmful/harmless prompt contrast pairs for steer calibration.

    Returns:
        (harmful, harmless) — prompt string lists of length n.
    """
    return refusal_prompt_pairs_large(n)
