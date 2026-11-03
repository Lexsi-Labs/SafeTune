"""Audit H10 (review of #111): AsFT must say when only some modules have an alignment direction.

q_proj has W_aligned == W_base (zero direction) and v_proj does not: the q_proj adapters were
skipped silently. One warning now gives the regularised and skipped counts and the first names.
"""
import logging

from test_H10_asft import _run, checkpoints, tok  # noqa: F401  (fixtures)


def test_partial_zero_direction_is_logged_once(monkeypatch, checkpoints, tok, caplog):  # noqa: F811
    tmp_path, base, aligned = checkpoints
    base_sd = base.state_dict()
    with __import__("torch").no_grad():
        for n, p in aligned.named_parameters():
            if n.endswith("q_proj.weight"):
                p.copy_(base_sd[n])                       # q_proj: zero direction
    n_q = sum(1 for n, _ in aligned.named_parameters() if n.endswith("q_proj.weight"))
    n_v = sum(1 for n, _ in aligned.named_parameters() if n.endswith("v_proj.weight"))
    assert n_q and n_v

    with caplog.at_level(logging.WARNING):
        tr, _ = _run(monkeypatch, tmp_path, tok, aligned, base_model_path=str(tmp_path / "base"))

    msgs = [r.getMessage() for r in caplog.records if "zero alignment direction" in r.getMessage()]
    assert len(msgs) == 1, msgs
    # the fixture shifts only q/v, so only v_proj has a direction; all 7 linears per layer are
    # LoRA-wrapped (q, k, v, o, gate, up, down) and the other 6 per layer are skipped
    assert f"regularising {n_v} LoRA modules" in msgs[0]
    assert f"skipped {7 * n_q - n_v} " in msgs[0]
    assert "q_proj" in msgs[0] and "v_proj" not in msgs[0].split("e.g.")[1]
    assert tr._use_lora_path and len(tr._proj_matrices) == n_v


def test_module_missing_from_the_base_state_dict_is_counted(checkpoints, tok, caplog):  # noqa: F811
    # Review of #111: a base whose keys do not match (here q_proj is absent) was skipped silently.
    from peft import LoraConfig, get_peft_model

    import safetune.harden as HARD
    from test_H10_asft import _data

    tmp_path, base, aligned = checkpoints
    model = get_peft_model(aligned, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                               task_type="CAUSAL_LM"))
    aligned_sd = {k: v.clone() for k, v in aligned.state_dict().items()}
    base_sd = {k: v for k, v in base.state_dict().items() if "q_proj" not in k}
    n_v = sum(1 for n, _ in model.named_parameters() if "v_proj.lora_A" in n)
    args = HARD.AsFTConfig(output_dir=str(tmp_path / "out"), max_steps=1, report_to=[], use_cpu=True,
                           save_strategy="no", per_device_train_batch_size=4)
    tr = HARD.AsFTHFTrainer(model=model, args=args, train_dataset=_data(tok),
                            aligned_state_dict=aligned_sd, base_state_dict=base_sd)
    with caplog.at_level(logging.WARNING):
        tr.train()
    assert len(tr._proj_matrices) == n_v
    msgs = [r.getMessage() for r in caplog.records if "LoRA modules" in r.getMessage()]
    assert len(msgs) == 1, msgs
    assert f"regularising {n_v} LoRA modules" in msgs[0]
    assert f"{n_v} with no matching weight" in msgs[0] and "q_proj" in msgs[0]
