"""Hackathon fixes for Cohere models (Tiny Aya validation report). CPU only, no
downloads: tokenizers are built in memory, models are tiny random configs."""
import pytest


def _word_tok(preamble_words: int = 0, bos_in_template: bool = True):
    """Whitespace word-level tokenizer that adds BOS itself (like Llama / Cohere)
    and whose chat template renders BOS plus a ``preamble_words``-token system
    preamble (Tiny Aya's template adds ~366 tokens)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = ["<unk>", "<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "pre"]
    words += "how do i bake bread mix flour and water what is the capital of france paris".split()
    vocab = {w: i for i, w in enumerate(words)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<bos>", eos_token="<eos>",
                                  pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = (("{{ bos_token }} " if bos_in_template else "")
                         + "pre " * preamble_words
                         + "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                           "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    return tok


_ROWS = [("how do i bake bread", "mix flour and water"),
         ("what is the capital of france", "paris")]


# ── C1: zero supervised tokens / max_len sizing ───────────────────────────────

def test_explicit_max_len_filled_by_preamble_raises():
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    tok = _word_tok(preamble_words=300)
    with pytest.raises(ValueError, match=r"max_len=256.*templated prompt is up to 30\d tokens"):
        _tokenize_qa_rows(tok, _ROWS, 256)


def test_some_rows_fully_masked_warns_with_count():
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    tok = _word_tok(preamble_words=10)
    long_row = (" ".join(["paris"] * 40), "paris")
    with pytest.warns(UserWarning, match="1 of 3 rows have no supervised token"):
        out = _tokenize_qa_rows(tok, _ROWS + [long_row], 32)
    assert all(len(r["input_ids"]) == 32 for r in out)  # explicit max_len kept exactly


def test_default_max_len_is_sized_from_the_templated_prompt():
    from datasets import Dataset
    from safetune.runner.utils.data_utils import tokenize_dataset
    tok = _word_tok(preamble_words=366)
    ds = Dataset.from_list([{"prompt": p, "response": r} for p, r in _ROWS])
    out = tokenize_dataset(ds, tok)  # max_len left at its default (None)
    longest_prompt = 1 + 366 + 1 + 6 + 1  # bos, preamble, <|user|>, words, <|assistant|>
    assert len(out[0]["input_ids"]) == longest_prompt + 256
    for row in out:
        assert any(l != -100 for l in row["labels"])


def test_auto_max_len_is_capped_and_floored():
    from safetune.runner.utils.data_utils import _resolve_max_len
    assert _resolve_max_len(_word_tok(0), _ROWS) == 1 + 8 + 256
    assert _resolve_max_len(_word_tok(0), _ROWS, response_budget=0) == 256  # floor
    assert _resolve_max_len(_word_tok(3000), _ROWS) == 2048
    assert _resolve_max_len(_word_tok(3000), _ROWS, max_len_cap=4096) == 1 + 3000 + 8 + 256
    assert _resolve_max_len(_word_tok(3000), _ROWS, 128) == 128  # explicit wins


def test_chat_template_text_gets_a_single_bos():
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    tok = _word_tok(preamble_words=0)
    row = _tokenize_qa_rows(tok, _ROWS[:1], 32)[0]
    ids = [i for i, m in zip(row["input_ids"], row["attention_mask"]) if m]
    assert ids.count(tok.bos_token_id) == 1
    sup = tok.decode([t for t in row["labels"] if t != -100])
    assert sup.split() == "mix flour and water".split()  # boundary not shifted by a BOS


# ── C2: refusal-direction layer selection ─────────────────────────────────────

def _sweep(monkeypatch, bypass_by_layer, baseline=0.9, n_layers=36, kl_by_layer=None,
           **cfg_kw):
    """Run the sweep with the generation / KL scorers stubbed out."""
    import torch
    import safetune.interventions.steer.refusal_direction as rd
    monkeypatch.setattr(rd, "_refusal_rate", lambda *a, **k: baseline)
    monkeypatch.setattr(rd, "_refusal_rate_when_ablated",
                        lambda m, t, vec, *a: bypass_by_layer[int(vec[0])])
    monkeypatch.setattr(rd, "_kl_when_ablated",
                        lambda m, t, vec, *a, **k: (kl_by_layer or {}).get(int(vec[0]), 0.01))
    dirs = {i: torch.full((4,), float(i)) for i in range(n_layers)}
    return rd._select_direction_by_scoring(
        None, None, dirs, ["h"], ["b"], n_layers=n_layers,
        cfg=rd.RefusalDirectionConfig(**cfg_kw))


def test_early_layer_winner_falls_back_to_middle(monkeypatch, caplog):
    bypass = {i: 0.5 for i in range(36)}
    bypass[1] = 0.0  # Tiny Aya: the sweep picked layer 1
    with caplog.at_level("WARNING"):
        assert _sweep(monkeypatch, bypass) is None
    assert "winner layer 1" in caplog.text
    assert _sweep(monkeypatch, bypass, min_layer_fraction=0.0) == 1  # overridable


def test_no_layer_helps_falls_back_to_middle(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        assert _sweep(monkeypatch, {i: 0.9 for i in range(36)}, baseline=0.9) is None
    assert "no layer cuts the clean refusal rate" in caplog.text


def test_ties_break_toward_the_middle_layer(monkeypatch, caplog):
    bypass = {i: 0.2 for i in range(36)}  # every layer ties
    with caplog.at_level("INFO"):
        assert _sweep(monkeypatch, bypass) == 18
    assert "selected layer 18 (bypass=0.200, kl=0.0100" in caplog.text


# ── C3 / C4: ReSta guidance for Tiny Aya (docs only) ──────────────────────────

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
_RESTA_DOC = _ROOT / "docs/user-guide/recover/layer/resta.md"


def test_resta_alpha_guidance_is_documented():
    import json
    from safetune.runner.recover import ReStaTrainer
    guidance = "On Tiny Aya, α=1 breaks the model; use α≈0.25"
    assert guidance in _RESTA_DOC.read_text()
    nb = json.loads((_ROOT / "examples/notebooks/recover_demo.ipynb").read_text())
    assert any(guidance in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "alpha~0.25" in ReStaTrainer.__doc__


def test_resta_memory_and_hackathon_notes_are_documented():
    from safetune.runner.recover import ReStaTrainer
    assert "one tensor at a time" in _RESTA_DOC.read_text() and '`device="cpu"`' in _RESTA_DOC.read_text()
    assert "one tensor at a time" in ReStaTrainer.__doc__ and 'device="cpu"' in ReStaTrainer.__doc__
    readme = (_ROOT / "README.md").read_text()
    notes = readme.split("## Cohere / hackathon notes", 1)[1].split("\n## ", 1)[0]
    for needle in ("max_len>=512", "pip uninstall -y torchao", "pick_layer=24", "Tiny Aya (3.35B)"):
        assert needle in notes, needle


def test_good_layers_just_over_kl_threshold_still_win(monkeypatch, caplog):
    """Qwen2.5-0.5B on GPU: layers 12-18 remove refusal but have KL 0.11-0.19;
    the only layer under 0.1 (layer 9) makes refusal *worse* than clean.
    The sweep must pick a good layer, not fall back to the middle."""
    n = 24
    bypass = {i: 0.75 for i in range(n)}
    kl = {i: 0.5 for i in range(n)}
    bypass[9], kl[9] = 0.812, 0.082          # under the threshold, worse than clean
    for layer, (b, k) in {12: (0.125, 0.11), 14: (0.0, 0.15), 16: (0.062, 0.12),
                          18: (0.0, 0.19)}.items():
        bypass[layer], kl[layer] = b, k
    with caplog.at_level("WARNING"):
        picked = _sweep(monkeypatch, bypass, baseline=0.688, n_layers=n, kl_by_layer=kl)
    assert picked == 14                       # lowest bypass; 14 and 18 tie, 14 nearer the middle
    assert "kl_threshold_fallback" in caplog.text
    # Disabling the fallback restores the strict paper rule (middle-layer fallback).
    assert _sweep(monkeypatch, bypass, baseline=0.688, n_layers=n, kl_by_layer=kl,
                  kl_threshold_fallback=None) is None


def test_under_threshold_layer_that_helps_still_preferred(monkeypatch):
    n = 24
    bypass = {i: 0.75 for i in range(n)}
    kl = {i: 0.5 for i in range(n)}
    bypass[10], kl[10] = 0.3, 0.05            # halves refusal, under threshold
    bypass[14], kl[14] = 0.0, 0.15            # better, but over threshold
    assert _sweep(monkeypatch, bypass, baseline=0.688, n_layers=n, kl_by_layer=kl) == 10


def test_marginal_under_threshold_layer_does_not_beat_a_real_one(monkeypatch):
    """Real Qwen2.5-0.5B on MPS: layer 10 (kl 0.07) only moves refusal 0.56 -> 0.50;
    layer 16 (kl 0.54) takes it to 0.06. Ablating layer 10 left 11/16 refusals."""
    n = 24
    bypass = {i: 0.75 for i in range(n)}
    kl = {i: 0.5 for i in range(n)}
    bypass[10], kl[10] = 0.500, 0.0705
    bypass[16], kl[16] = 0.062, 0.5355
    assert _sweep(monkeypatch, bypass, baseline=0.562, n_layers=n, kl_by_layer=kl) == 16
    assert _sweep(monkeypatch, bypass, baseline=0.562, n_layers=n, kl_by_layer=kl,
                  min_bypass_reduction=0.0) == 10  # old behaviour, still selectable
