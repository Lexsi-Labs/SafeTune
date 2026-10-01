from typing import Optional
from safetune.runner.utils.data_utils import harden_contamination_sets, build_sft_dataset


def load_harden_dataset(tok, *, n: int = 256, max_len: Optional[int] = None,
                        harmful_frac: float = 0.15):
    """Contaminated SFT train set and refusal safety set for harden training.

    Returns:
        (train_ds, safety_ds) — tokenized HF Datasets.
    """
    contamination, _, refusal_aux = harden_contamination_sets(tok, n=n, max_len=max_len)
    train_ds = build_sft_dataset(
        tok, n_total=n, harmful_frac=harmful_frac, harmful_ds=contamination, max_len=max_len
    )
    return train_ds, refusal_aux
