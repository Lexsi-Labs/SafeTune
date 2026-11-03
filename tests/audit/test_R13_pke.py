"""Audit R13: PKE's locator ranks rows across every tensor in a layer but edits down_proj rows.

The edit only touches mlp.down_proj rows (PKE/DINM edit the located layer's down_proj), and the
docstring says the located region is the top-k down_proj rows that drifted most. The locator pools
the rows of q/k/v/o/gate/up/down and norms of a layer and keeps the top-k row *indices*, so a large
change in another tensor's row i selects down_proj row i, even when down_proj did not change there.
"""
import copy
import logging

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from safetune.interventions.recover.pke import PKEConfig
from safetune.core.pke import ToxicNeuronLocator


def test_pke_locator_ranks_down_proj_rows_only():
    torch.manual_seed(0)
    clean = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=1, num_attention_heads=2,
                                         num_key_value_heads=2))
    toxic = copy.deepcopy(clean)
    with torch.no_grad():
        toxic.model.layers[0].mlp.down_proj.weight[2] += 0.1        # the only down_proj change
        toxic.model.layers[0].self_attn.q_proj.weight[7] += 5.0     # a bigger change elsewhere

    located = ToxicNeuronLocator(config=PKEConfig(top_k_neurons=1)).locate_by_weight_change(
        clean.state_dict(), toxic.state_dict())

    assert located[0] == [2], f"located rows {located[0]} (row 7 comes from q_proj, not down_proj)"


# --- review of #109: LoRA keys are not down_proj rows -------------------------------------

def _wrapped(change_lora_a: bool):
    torch.manual_seed(0)
    p = "model.layers.0.mlp.down_proj."
    sd = {p + "base_layer.weight": torch.zeros(8, 16),
          p + "lora_A.default.weight": torch.zeros(4, 16),     # rows indexed by LoRA rank
          p + "lora_B.default.weight": torch.zeros(8, 4)}
    toxic = {k: v.clone() for k, v in sd.items()}
    if change_lora_a:
        toxic[p + "lora_A.default.weight"][1] += 5.0            # rank index 1, not neuron 1
    return sd, toxic


def test_lora_keys_are_not_located_as_down_proj_rows(caplog):
    clean, toxic = _wrapped(change_lora_a=True)
    with caplog.at_level(logging.WARNING, logger="safetune.core.pke"):
        located = ToxicNeuronLocator(config=PKEConfig(top_k_neurons=1)).locate_by_weight_change(
            clean, toxic)
    # base_layer is unchanged, so no row has a real signal; row 1 is only LoRA rank index 1.
    assert 1 not in located.get(0, []), f"located {located} from a LoRA-rank row"
    # Nothing moved in down_proj, so nothing is located (not an arbitrary row), and the
    # warning says why.
    assert located == {}, f"located {located} from all-equal deltas"
    assert "LoRA" in caplog.text and "no rows were located" in caplog.text
    assert "effectively identical" not in caplog.text


def test_adapter_only_state_dict_warns_that_nothing_was_located(caplog):
    p = "model.layers.0.mlp.down_proj."
    clean = {p + "lora_A.default.weight": torch.zeros(4, 16)}
    toxic = {p + "lora_A.default.weight": torch.ones(4, 16)}
    with caplog.at_level(logging.WARNING, logger="safetune.core.pke"):
        located = ToxicNeuronLocator(config=PKEConfig(top_k_neurons=1)).locate_by_weight_change(
            clean, toxic)
    assert located == {}
    assert "LoRA" in caplog.text


def test_plain_and_base_layer_keys_still_located(caplog):
    p = "model.layers.0.mlp.down_proj."
    for key in (p + "weight", p + "base_layer.weight"):
        clean = {key: torch.zeros(8, 16)}
        toxic = {key: torch.zeros(8, 16)}
        toxic[key][3] += 1.0
        with caplog.at_level(logging.WARNING, logger="safetune.core.pke"):
            located = ToxicNeuronLocator(config=PKEConfig(top_k_neurons=1)).locate_by_weight_change(
                clean, toxic)
        assert located == {0: [3]}, (key, located)
    assert not caplog.records


def test_apply_pke_edits_nothing_when_no_down_proj_row_moved():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2)
    clean = LlamaForCausalLM(cfg)
    toxic = copy.deepcopy(clean)
    model = copy.deepcopy(clean)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kw):
            ids = torch.tensor([[2, 3, 4, 5]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    located = ToxicNeuronLocator(config=PKEConfig(top_k_neurons=1)).locate_by_weight_change(
        clean.state_dict(), toxic.state_dict())
    assert located == {}
    from safetune.interventions.recover.pke import PKEGradientEditor
    PKEGradientEditor(model=model, toxic_neurons=located, tokenizer=_Tok(),
                      config=PKEConfig(num_steps=2)).apply_edits(
        harmful_prompt="x", safe_response="y", locality_inputs=["z"])
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed"
