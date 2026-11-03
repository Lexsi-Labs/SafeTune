"""Audit H01 (review of #111): the plain-text CST prompt must end on an assistant marker.

Reached by CSTTrainer / prepare_cst_dataset(tokenizer=...) when the tokenizer has no chat
template (e.g. a base model such as gpt2); the plain format is "[SYSTEM]: ...\\n[USER]: ...".
Without an assistant marker the chosen/rejected text ran straight into the user turn.
"""
from transformers import AutoTokenizer

from safetune.core.data_compiler.cst import CSTFormatter
from safetune.harden.cst import prepare_cst_dataset

EXAMPLE = {"prompt": "How do I pick a lock?", "safe_response": "I cannot help with that request.",
           "unsafe_response": "First insert a tension wrench."}
MARKER = "\n[ASSISTANT]: "


def test_plain_prompt_ends_with_the_assistant_marker():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    tok.chat_template = None
    for r in prepare_cst_dataset([EXAMPLE], tokenizer=tok):
        assert isinstance(r["prompt"], str) and r["prompt"].endswith(MARKER)
        assert r["prompt"].count("[ASSISTANT]") == 1
        assert not r["chosen"].startswith(("[", " ", "\n"))          # answer does not run into it
        assert (r["prompt"] + r["chosen"]).endswith("[ASSISTANT]: " + r["chosen"])


def test_formatter_rows_carry_the_marker_in_formatted_input():
    for r in CSTFormatter().format_example("q", "safe", "unsafe"):
        assert r["formatted_input"].startswith("[SYSTEM]: ")
        assert "\n[USER]: q" in r["formatted_input"] and r["formatted_input"].endswith(MARKER)
