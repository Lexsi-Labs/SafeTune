"""Regressions for the SafeGrad fp16 scale, the safety dataset labels and left padding."""
import pytest
import torch

_ROWS = [("how do i bake bread", "mix flour and water"),
         ("what is the capital of france", "paris")]


def _word_tok(preamble_words=0):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = ["<unk>", "<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "pre"]
    words += "how do i bake bread mix flour and water what is the capital of france paris".split()
    t = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<bos>", eos_token="<eos>",
                                  pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = ("{{ bos_token }} " + "pre " * preamble_words
                         + "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                           "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    return tok


def _supervised(row):
    return [t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]


@pytest.mark.parametrize("side", ["right", "left"])
def test_only_the_response_is_supervised_whatever_the_padding_side(side):
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    tok = _word_tok(preamble_words=3)
    tok.padding_side = side
    row = _tokenize_qa_rows(tok, _ROWS[:1], 40)[0]
    # "mix flour and water" + the template's trailing space; nothing from the prompt
    resp = tok("mix flour and water", add_special_tokens=False)["input_ids"]
    assert _supervised(row)[:len(resp)] == resp
    assert row["labels"][:row["attention_mask"].index(1)] == [-100] * row["attention_mask"].index(1)


def test_left_and_right_padding_supervise_the_same_tokens():
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    got = {}
    for side in ("right", "left"):
        tok = _word_tok(preamble_words=3)
        tok.padding_side = side
        got[side] = _supervised(_tokenize_qa_rows(tok, _ROWS[:1], 40)[0])
    assert got["right"] == got["left"] and got["right"]


def test_safety_dataset_supervises_the_refusal_not_the_prompt(monkeypatch):
    from safetune.runner.utils import data_utils
    tok = _word_tok(preamble_words=3)
    monkeypatch.setattr(data_utils, "_REFUSAL_QA", [_ROWS[0]])
    monkeypatch.setattr(data_utils, "spec", lambda name: {})
    row = data_utils.build_safety_dataset(tok, max_len=40)[0]
    resp = tok("mix flour and water", add_special_tokens=False)["input_ids"]
    assert _supervised(row)[:len(resp)] == resp
    prompt = tok("how do i bake bread", add_special_tokens=False)["input_ids"]
    assert not set(prompt) & set(_supervised(row))


class _Scaler:
    def __init__(self, scale): self.scale = scale
    def is_enabled(self): return self.scale is not None
    def get_scale(self): return self.scale


def _surgery(scale):
    from safetune.interventions.harden.safegrad import SafeGradHFTrainer
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 3, bias=False)
    g_user = torch.randn(3, 4)
    g_align = -g_user + 0.3 * torch.randn(3, 4)          # conflicting, so the projection runs
    tr = object.__new__(SafeGradHFTrainer)
    tr._rho = 1.0
    tr._trainable_params = lambda model: [lin.weight]
    tr.accelerator = type("A", (), {"scaler": _Scaler(scale)})()
    s = 1.0 if scale is None else scale
    lin.weight.grad = (g_user * s).clone()               # what loss.backward() leaves under the GradScaler
    tr._apply_global_surgery(lin, [g_align.clone()])
    return lin.weight.grad / s                           # what the optimizer sees after unscaling


def test_safegrad_update_does_not_depend_on_the_fp16_loss_scale():
    base = _surgery(None)
    scaled = _surgery(65536.0)
    assert torch.allclose(base, scaled, rtol=1e-4, atol=1e-6)


def test_a_result_file_full_of_newlines_is_rejected_quickly(monkeypatch):
    """The reader must not re-split the buffer once per line: 16 MiB of newlines would take hours."""
    import time
    from safetune.rewards import _code_grader as cg
    monkeypatch.setattr(
        cg, "_HARNESS_MANY",
        "import sys\nopen(sys.argv[2], 'wb').write(b'\\n' * int(sys.argv[4]))\n")
    t0 = time.monotonic()
    got = cg.grade_assertions("x = 1", ["assert x == 1", "assert x == 1"], 5.0)
    assert time.monotonic() - t0 < 20
    assert got == [False, False]


def test_antibody_turns_fp16_off_given_args_positionally_or_by_keyword(tmp_path):
    from transformers import TrainingArguments
    from safetune.interventions.harden.antibody import AntibodyTrainer
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2)
    seen = []
    import transformers
    orig = transformers.Trainer.__init__
    monkey = lambda self, *a, **k: seen.append((a, k))  # noqa: E731
    transformers.Trainer.__init__ = monkey
    try:
        args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
        object.__setattr__(args, "fp16", True)
        for call in (lambda: AntibodyTrainer(LlamaForCausalLM(cfg), args), lambda: AntibodyTrainer(model=LlamaForCausalLM(cfg), args=args)):
            try:
                call()
            except Exception:
                pass
    finally:
        transformers.Trainer.__init__ = orig
    got = [(a[1] if len(a) > 1 else k["args"]).fp16 for a, k in seen]
    assert got == [False, False], got
