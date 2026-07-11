from safetune.runner.utils.data_utils import unlearn_forget_retain


def load_unlearn_dataset(tok, *, n: int = 256, max_len: int = 256):
    """Forget/retain tokenized datasets for unlearn training.

    Returns:
        (forget_ds, retain_ds) — tokenized HF Datasets.
    """
    return unlearn_forget_retain(tok, n=n, max_len=max_len)
