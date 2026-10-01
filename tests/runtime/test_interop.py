"""Interop with the Lexsi stack (Track 2: CuratorKIT -> SafeTune -> CircuitKIT).

CPU only, no downloads beyond the tiny Llama tokenizer the other runtime tests use;
models are tiny random ``cohere`` / ``aya_vision`` built from config classes.

Covers: a CuratorKIT export folder (built here with the same layout: README
``configs:``, manifest / rejected / provenance files) loads per config; SafeTune
tokenises chat ``messages``, ShareGPT ``conversations``, Alpaca with ``input`` and
DPO rows itself; prompt-only rows raise; CircuitKIT ``*_scores.json`` (old
``node_scores`` only and the 0.2 ``safety_units`` / ``layer_suggestions`` keys);
``lexsi_provenance.json`` in checkpoint, results and steering-vector dirs;
``push_to_hub`` (mocked); the version; and the Track-2 chain
CuratorKIT folder -> harden 1 step -> a folder the transformers Auto classes load,
with provenance that chains to the dataset's.
"""
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path

import pytest
import torch

from .test_vision_archs import TINY_LLAMA_TOK, _model

# A fixed template so the label checks do not depend on the tokenizer's own.
_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}\n{% endfor %}"
             "{% if add_generation_prompt %}<|assistant|>{% endif %}")

CURATOR_PROV = {"schema": "lexsi.provenance/1", "library": "curatorkit", "version": "0.4.0",
                "git_sha": None, "created_at": "2026-09-28T00:00:00Z", "base_model": None,
                "method": "curate", "inputs": [], "params": {}}

ROWS = {
    "sft_alpaca": [{"instruction": "Name the capital city.", "input": "Country: France",
                    "output": "Paris is the capital."}],
    "sft_sharegpt": [{"messages": [{"role": "system", "content": "Be brief."},
                                   {"role": "user", "content": "Name the capital of France."},
                                   {"role": "assistant", "content": "Paris is the capital."}]}],
    "dpo": [{"prompt": "Name the capital of France.", "chosen": "Paris is the capital.",
             "rejected": "Berlin."}],
    "grpo": [{"prompt": "Name the capital of France.", "responses": ["Paris", "Rome"],
              "rewards": [1.0, 0.0]}],
    "ppo": [{"prompt": "Name the capital of France."}],
}


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        t = AutoTokenizer.from_pretrained(TINY_LLAMA_TOK)
    except Exception as e:  # offline
        pytest.skip(f"tokenizer {TINY_LLAMA_TOK} unavailable: {e}")
    t.pad_token = t.pad_token or t.eos_token
    t.padding_side = "right"
    t.chat_template = _TEMPLATE
    return t


def curator_folder(root: Path, rows=ROWS, splits=("train",)) -> Path:
    """The folder CuratorKIT 0.4 exports (HK-CU/interop): one jsonl per format and
    split, a README whose YAML ``configs:`` names them, plus non-data files."""
    root.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex  # datasets caches a folder by its README configs
    lines = ["---", "configs:"]
    for i, (cfg, rs) in enumerate(rows.items()):
        lines += [f"- config_name: {cfg}"] + (["  default: true"] if i == 0 else [])
        lines += [f"  description: curatorkit {cfg} export, sha256 {tag}", "  data_files:"]
        for split in splits:
            path = (root / split / f"{cfg}.jsonl") if len(splits) > 1 else root / f"{cfg}.jsonl"
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in rs))
            lines += [f"  - split: {split}", f"    path: {path.relative_to(root).as_posix()}"]
    (root / "README.md").write_text("\n".join(lines + ["---", "", "# card", ""]))
    (root / "manifest.json").write_text(json.dumps({"provenance": CURATOR_PROV, "rows": 5}))
    (root / "rejected.jsonl").write_text(json.dumps({"reason": "dup", "text": "x"}) + "\n")
    (root / "lexsi_provenance.json").write_text(json.dumps(CURATOR_PROV))
    return root


def _supervised(tok, row):
    return tok.decode([i for i, l in zip(row["input_ids"], row["labels"]) if l != -100])


# ── 1. Datasets in ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("splits", [("train",), ("train", "validation")])
def test_curatorkit_folder_loads_per_config(tmp_path, splits):
    from safetune.data.dataset_ids import load
    root = curator_folder(tmp_path / "out", splits=splits)
    for cfg, rows in ROWS.items():
        for split in splits:
            ds = load(str(root), config=cfg, split=split)
            assert ds.num_rows == len(rows)
            assert set(ds.column_names) == set(rows[0])  # never manifest / rejected columns


def test_folder_without_card_never_picks_a_file_silently(tmp_path):
    from safetune.data.dataset_ids import load
    for name in ("a", "b"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    with pytest.raises(ValueError, match="not found"):
        load(str(tmp_path))
    assert load(str(tmp_path / "a.jsonl")).num_rows == 1


@pytest.mark.parametrize("cfg", ["sft_alpaca", "sft_sharegpt", "dpo"])
def test_formats_are_tokenised_with_loss_on_the_response(tmp_path, tok, cfg):
    from safetune.data.dataset_ids import load
    from safetune.runner.utils.data_utils import tokenize_dataset
    out = tokenize_dataset(load(str(curator_folder(tmp_path / "o")), config=cfg), tok, max_len=64)
    assert out.num_rows == 1 and {"input_ids", "attention_mask", "labels"} <= set(out.column_names)
    row = out[0]
    assert "Paris" in _supervised(tok, row) and "France" not in _supervised(tok, row)
    text = tok.decode(row["input_ids"])
    if cfg == "sft_alpaca":
        assert "Country: France" in text  # Alpaca input is kept
    if cfg == "sft_sharegpt":
        assert "Be brief." in text  # system turn is kept


def test_sharegpt_conversations_and_turn_lists(tok):
    from datasets import Dataset
    from safetune.runner.utils.data_utils import tokenize_dataset
    conv = Dataset.from_list([{"conversations": [
        {"from": "human", "value": "Hi there"}, {"from": "gpt", "value": "Hello friend"},
        {"from": "human", "value": "Name the capital of France."},
        {"from": "gpt", "value": "Paris is the capital."}]}])
    sup = _supervised(tok, tokenize_dataset(conv, tok, max_len=96)[0])
    assert "Paris" in sup and "Hello" not in sup  # loss on the last assistant turn
    turns = Dataset.from_list([{  # DPO implicit: chosen repeats the prompt turns
        "prompt": [{"role": "user", "content": "Name the capital of France."}],
        "chosen": [{"role": "user", "content": "Name the capital of France."},
                   {"role": "assistant", "content": "Paris is the capital."}]}])
    row = tokenize_dataset(turns, tok, max_len=64)[0]
    assert "Paris" in _supervised(tok, row)
    assert tok.decode(row["input_ids"]).count("France") == 1


@pytest.mark.parametrize("cfg", ["ppo", "grpo"])
def test_prompt_only_rows_raise(tmp_path, tok, cfg):
    from safetune.data.dataset_ids import load
    from safetune.runner.utils.data_utils import tokenize_dataset
    with pytest.raises(ValueError, match="nothing to fine-tune"):
        tokenize_dataset(load(str(curator_folder(tmp_path / "o")), config=cfg), tok)


def test_rows_without_response_are_skipped_with_a_warning(tok):
    from datasets import Dataset
    from safetune.runner.utils.data_utils import tokenize_dataset
    ds = Dataset.from_list([{"prompt": "Name the capital of France.", "response": "Paris."},
                            {"prompt": "Say nothing.", "response": "  "}])
    with pytest.warns(UserWarning, match="1 of 2 training rows"):
        assert tokenize_dataset(ds, tok, max_len=64).num_rows == 1


def test_cli_rejects_prompt_only_train_data(tmp_path, tok, monkeypatch):
    """Was: `--train-dataset ppo.jsonl` fine-tuned on empty responses, no warning."""
    from safetune import cli
    monkeypatch.setattr(cli, "_load_model_and_tok", lambda path: (_model("cohere"), tok))
    monkeypatch.setattr(sys, "argv", [
        "safetune", "train", "--model", "tiny", "--algo", "plainsft",
        "--train-dataset", str(curator_folder(tmp_path / "o")), "--train-config", "ppo",
        "--output", str(tmp_path / "ckpt")])
    with pytest.raises(ValueError, match="nothing to fine-tune"):
        cli.main()


def test_trainer_takes_a_raw_dataset(tok):
    """CuratorKIT's README hands SafeTune `load_dataset(dir, cfg, split="train")`."""
    import inspect
    from datasets import Dataset
    from safetune.runner.harden import PlainSFTTrainer, SafeGradTrainer
    seen = {}

    class Probe(PlainSFTTrainer):  # every subclass's train() gets the wrapper
        def train(self, train_dataset, out_dir=None, *, safety_dataset=None, **kw):
            seen.update(train=train_dataset.column_names, safety=safety_dataset.column_names)

    Probe(_model("cohere"), tok).train(Dataset.from_list(ROWS["sft_sharegpt"]),
                                       safety_dataset=Dataset.from_list(ROWS["dpo"]))
    assert "input_ids" in seen["train"] and "input_ids" in seen["safety"]
    # The CLI reads train()'s signature to decide whether --safety-dataset applies.
    assert "safety_dataset" in inspect.signature(SafeGradTrainer(_model("cohere"), tok).train).parameters


# ── 2. CircuitKIT handoff ────────────────────────────────────────────────────

NODE_SCORES = {"A0.1": 0.9, "A1.3": 0.2, "MLP 1": 1.0, "L0.7": 0.1, "A0.0": 0.05}


def test_circuitkit_node_scores_only(tmp_path):
    """Was: an empty CircuitInfo (every field None) for CircuitKIT's own format."""
    from safetune.core.circuit_kit import load_circuit_info_from_file
    path = tmp_path / "safety_scores.json"
    path.write_text(json.dumps({"node_scores": NODE_SCORES, "model": "tiny"}))
    info = load_circuit_info_from_file(str(path), top_fraction=0.4)
    su, ls = info.safety_units, info.layer_suggestions
    assert su.unit_ids == ["MLP 1", "A0.1"]
    assert su.module_names == ["model.layers.0.self_attn", "model.layers.1.mlp"]
    assert su.activation_correlation == {"MLP 1": 1.0, "A0.1": 0.9}
    assert ls.layer_subset == [0, 1]
    assert set(ls.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"}


def test_circuitkit_02_scores_file(tmp_path):
    """CircuitKIT 0.2 (TL3 branch) writes the SafeTune keys next to node_scores."""
    from safetune.core.circuit_kit import load_circuit_info_from_file
    from safetune.core.safety_lora import build_safety_lora_config
    data = {"node_scores": NODE_SCORES,
            "safety_units": {"layer_indices": [3], "module_names": ["model.layers.3.self_attn"],
                             "unit_ids": ["A3.1"], "activation_correlation": {"A3.1": 1.0}},
            "layer_suggestions": {"target_modules": ["q_proj", "v_proj"], "layer_subset": [3],
                                  "priority": {"q_proj": 1.0, "v_proj": 1.0}},
            "provenance": {"schema": "lexsi.provenance/1", "library": "circuitkit"}}
    path = tmp_path / "x_scores.json"
    path.write_text(json.dumps(data))
    info = load_circuit_info_from_file(str(path))
    assert info.safety_units.layer_indices == [3]  # the file's keys win over node_scores
    assert info.layer_suggestions.target_modules == ["q_proj", "v_proj"]
    cfg = build_safety_lora_config(circuit_guided=True, circuit_guide_path=str(path))
    assert cfg.circuit_info.layer_suggestions.layer_subset == [3]


@pytest.mark.parametrize("data,match", [({"scores": {}}, "found keys"),
                                        ({"node_scores": {"blocks.0.hook_z": 1.0}},
                                         "recognised name")])
def test_circuitkit_without_circuit_raises(tmp_path, data, match):
    from safetune.core.circuit_kit import load_circuit_info_from_file
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_circuit_info_from_file(str(path))


# ── 3. Provenance ────────────────────────────────────────────────────────────

def _prov(path):
    return json.loads((Path(path) / "lexsi_provenance.json").read_text())


def _source(tmp_path, arch, tok, parent=None):
    """A saved tiny model folder (with a processor for aya_vision)."""
    import transformers as T
    src = tmp_path / f"src_{arch}"
    _model(arch).save_pretrained(src)
    tok.save_pretrained(src)
    if arch == "aya_vision":
        T.AyaVisionProcessor(
            image_processor=T.GotOcr2ImageProcessor(size={"height": 32, "width": 32},
                                                   crop_to_patches=False, max_patches=1),
            tokenizer=tok, patch_size=16, img_size=32, downsample_factor=2).save_pretrained(src)
    if parent:
        (src / "lexsi_provenance.json").write_text(json.dumps(parent))
    return src


def test_save_checkpoint_writes_provenance_with_the_model_lineage(tmp_path, tok):
    from safetune.runner.utils.model_utils import load_model, load_tok, save_checkpoint
    parent = {**CURATOR_PROV, "library": "aligntune", "base_model": "CohereLabs/aya-expanse-8b"}
    src = _source(tmp_path, "cohere", tok, parent=parent)
    path = save_checkpoint(load_model(str(src), dtype=torch.float32, device="cpu"),
                           load_tok(str(src), cache=False), "ck", out_dir=str(tmp_path),
                           method="recover.Test", params={"alpha": 0.5})
    p = _prov(path)
    assert p["schema"] == "lexsi.provenance/1" and p["library"] == "safetune"
    assert p["method"] == "recover.Test" and p["params"] == {"alpha": 0.5}
    assert p["inputs"][0] == {"kind": "model", "ref": str(src), "provenance": parent}
    assert p["base_model"] == "CohereLabs/aya-expanse-8b"  # follows the lineage


def test_results_and_vectors_dirs_get_provenance(tmp_path, tok):
    from safetune.core.runtime.inference.vector_extraction import (
        SteeringVectorExtractor, VectorExtractionConfig)
    from safetune.runner.utils.results_writer import ResultsWriter
    w = ResultsWriter("harden", results_dir=str(tmp_path / "res"))
    w.append({"method": "Lisa", "variant": "default", "metrics": {}})
    assert _prov(Path(w.path).parent)["method"] == "harden.Lisa"
    ex = SteeringVectorExtractor(_model("cohere"), tok, VectorExtractionConfig(target_layers=[0, 1]))
    (tmp_path / "vec").mkdir()
    ex.save_vectors({0: torch.zeros(4)}, str(tmp_path / "vec" / "v.pt"))
    p = _prov(tmp_path / "vec")
    assert p["method"] == "steer.extract_vectors" and p["params"]["file"] == "v.pt"


# ── 4. Hub push (mocked) ─────────────────────────────────────────────────────

def test_push_to_hub(tmp_path, monkeypatch):
    import huggingface_hub
    import safetune
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("init", token))

        def create_repo(self, repo_id, **kw):
            calls.append(("create_repo", repo_id, kw))

        def upload_folder(self, **kw):
            calls.append(("upload_folder", kw))
            return "commit"

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "lexsi_provenance.json").write_text("{}")
    (ck / "results_harden.json").write_text("[]")
    assert safetune.push_to_hub(str(ck), "org/m", private=True) == "commit"
    assert calls[1] == ("create_repo", "org/m",
                        {"repo_type": "model", "private": True, "exist_ok": True})
    assert calls[2][1]["folder_path"] == str(ck) and calls[2][1]["allow_patterns"] is None
    calls.clear()
    safetune.push_to_hub(str(ck / "results_harden.json"), "org/r", repo_type="dataset")
    assert calls[2][1]["allow_patterns"] == ["results_harden.json", "lexsi_provenance.json"]
    assert calls[2][1]["repo_type"] == "dataset"


# ── 5. Version ───────────────────────────────────────────────────────────────

def test_version_comes_from_metadata():
    import safetune
    from scripts.release import current_version

    # The authoritative version is read from the tree (CITATION.cff + pyproject.toml
    # agree, or current_version raises), not hardcoded here: the release pipeline
    # bumps the tree to the next patch version before this suite runs, so a
    # hardcoded literal failed every release build ('0.2.1' != '0.2.0').
    expected = current_version(Path(__file__).parents[2])
    try:
        installed = importlib.metadata.version("safetune")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("safetune is not installed")
    assert safetune.__version__ == installed == expected


# ── 6. Track 2 chain ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("arch", ["cohere", "aya_vision"])
@pytest.mark.parametrize("entry", ["python", "cli"])
def test_track2_chain(tmp_path, tok, arch, entry, monkeypatch):
    """CuratorKIT folder -> harden 1 step -> an HF folder the Auto classes load, with
    provenance that chains to the CuratorKIT export."""
    import transformers as T
    from safetune.runner.harden import PlainSFTTrainer
    from safetune.runner.utils.model_utils import load_model, load_tok
    data = curator_folder(tmp_path / "curated")
    src = _source(tmp_path, arch, tok)
    model = load_model(str(src), dtype=torch.float32, device="cpu")
    mtok = load_tok(str(src), cache=False)
    out = tmp_path / "hardened"
    if entry == "python":
        path = PlainSFTTrainer(model, mtok, epochs=1, batch_size=1).train(
            str(data), dataset_config="sft_sharegpt", out_dir=str(out))
    else:
        from safetune import cli
        monkeypatch.setattr(cli, "_load_model_and_tok", lambda p: (model, mtok))
        monkeypatch.setattr(sys, "argv", [
            "safetune", "train", "--model", str(src), "--algo", "plainsft",
            "--train-dataset", str(data), "--train-config", "sft_alpaca",
            "--precision", "fp32", "--output", str(out)])
        cli.main()
        path = str(out)
    auto = T.AutoModelForImageTextToText if arch == "aya_vision" else T.AutoModelForCausalLM
    assert type(auto.from_pretrained(path)).__name__ == type(model).__name__
    T.AutoTokenizer.from_pretrained(path)
    if arch == "aya_vision":
        assert type(T.AutoProcessor.from_pretrained(path)).__name__ == "AyaVisionProcessor"
    p = _prov(path)
    assert p["method"] == "harden.PlainSFT" and p["library"] == "safetune"
    model_in, data_in = p["inputs"][0], p["inputs"][1]
    assert model_in["kind"] == "model" and model_in["ref"] == str(src)
    assert data_in["kind"] == "dataset" and data_in["ref"] == str(data)
    assert data_in["config"] == ("sft_sharegpt" if entry == "python" else "sft_alpaca")
    assert data_in["provenance"]["library"] == "curatorkit"  # the parent chain
