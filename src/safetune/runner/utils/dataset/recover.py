from safetune.runner.utils.data_utils import recover_calib_input_ids


def load_recover_dataset(tok, device, *, max_len: int = 64) -> list:
    """Tokenized input_ids for recover calibration (harmful-first).

    Returns:
        list of (1, seq_len) tensors on device.
    """
    return recover_calib_input_ids(tok, device, max_len=max_len)
