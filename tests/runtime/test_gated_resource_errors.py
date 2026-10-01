"""A gated/inaccessible Hugging Face Hub model or dataset (walledai/HarmBench,
allenai/wildjailbreak, meta-llama/Llama-3.1-8B-Instruct as the hexphi/orbench/
ailuminate judge, ...) must fail evaluate() with a stable, catchable error --
not a raw huggingface_hub/OSError a caller has to string-match, and not vLLM's
own wrapper exception around the same root cause.

``GatedResourceError`` (error_code "GATED_RESOURCE", carrying ``repo_id`` and
``kind``) is what a caller like a training service or a UI should catch
instead. ``hf_access_errors()`` is the context manager that produces it at
every real load site: judge model loading (_load_hf_model, the vLLM judge
path) and dataset loading (dataset_ids.load()).
"""
import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from safetune.utils.errors import GatedResourceError, hf_access_errors


def _hub_error(cls, message):
    resp = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/x"))
    return cls(message, response=resp)


def test_gated_repo_error_is_translated_with_request_access_suggestion():
    with pytest.raises(GatedResourceError) as exc_info:
        with hf_access_errors("meta-llama/Llama-3.1-8B-Instruct", kind="judge model"):
            raise _hub_error(GatedRepoError, "403 Client Error")
    err = exc_info.value
    assert err.error_code == "GATED_RESOURCE"
    assert err.repo_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert err.kind == "judge model"
    assert any("Request access" in s for s in err.suggestions)
    assert "gated" in str(err)


def test_repository_not_found_error_is_translated_without_gated_wording():
    with pytest.raises(GatedResourceError) as exc_info:
        with hf_access_errors("bogus/does-not-exist", kind="dataset"):
            raise _hub_error(RepositoryNotFoundError, "404 Client Error")
    err = exc_info.value
    assert err.error_code == "GATED_RESOURCE"
    assert "could not be found or accessed" in str(err)
    assert not any("Request access" in s for s in err.suggestions)


def test_wrapped_hub_error_is_still_found_through_the_cause_chain():
    """vLLM (and sometimes transformers) re-raises the real huggingface_hub
    error wrapped in its own exception type -- a plain isinstance/except on
    the outer exception would miss it."""
    original = _hub_error(GatedRepoError, "403 Client Error")
    with pytest.raises(GatedResourceError) as exc_info:
        with hf_access_errors("cais/HarmBench-Mistral-7b-val-cls", kind="judge model"):
            try:
                raise original
            except GatedRepoError as e:
                raise RuntimeError("vLLM failed to load the model") from e
    assert exc_info.value.error_code == "GATED_RESOURCE"


def test_unrelated_errors_pass_through_unchanged():
    with pytest.raises(ValueError, match="unrelated"):
        with hf_access_errors("some/repo"):
            raise ValueError("unrelated failure")


def test_dataset_ids_load_raises_gated_resource_error(monkeypatch):
    import safetune.data.dataset_ids as ids

    def resolve(source, config_name=None, split=None, **kw):
        raise _hub_error(GatedRepoError, "403 Client Error")

    monkeypatch.setattr(ids.LoaderResolver, "resolve", staticmethod(resolve))
    with pytest.raises(GatedResourceError) as exc_info:
        ids.load("wildjailbreak")
    assert exc_info.value.kind == "dataset"
    assert exc_info.value.repo_id == "allenai/wildjailbreak"


def test_load_hf_model_raises_gated_resource_error(monkeypatch):
    import importlib  # the evaluate() path (module name is shadowed by the function)
    ev = importlib.import_module("safetune.instrumentation.evaluate.suite.evaluate")

    def from_pretrained(model_id, **kw):
        raise _hub_error(GatedRepoError, "403 Client Error")

    monkeypatch.setattr(ev.AutoTokenizer, "from_pretrained", staticmethod(from_pretrained))
    with pytest.raises(GatedResourceError) as exc_info:
        ev._load_hf_model("meta-llama/Llama-3.1-8B-Instruct")
    assert exc_info.value.kind == "judge model"
    assert exc_info.value.repo_id == "meta-llama/Llama-3.1-8B-Instruct"


def test_transformers_backend_raises_gated_resource_error(monkeypatch):
    """Generator, ASRT's attacker model and BoN red-teaming all construct a
    TransformersBackend -- this is the one choke point that covers all of
    them, uniformly with the judge/dataset loaders above.

    ``_ensure_loaded()`` does ``from transformers import AutoTokenizer``
    locally, so the real ``transformers.AutoTokenizer`` has to be patched,
    not a module-level name on the backend module.
    """
    import transformers
    from safetune.core.eval.pipeline.backends.transformers import TransformersBackend

    def from_pretrained(model_id, **kw):
        raise _hub_error(GatedRepoError, "403 Client Error")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(from_pretrained))
    backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
    with pytest.raises(GatedResourceError) as exc_info:
        backend._ensure_loaded()
    assert exc_info.value.kind == "model"
    assert exc_info.value.repo_id == "meta-llama/Llama-3.2-1B-Instruct"


def test_vllm_backend_raises_gated_resource_error(monkeypatch):
    """Same mechanism, vLLM path (only runs where vllm is installed -- a real
    GPU machine; this sandbox has no vllm). ``_ensure_loaded()`` does
    ``from vllm import LLM`` locally, so ``vllm.LLM`` itself is patched."""
    vllm = pytest.importorskip("vllm")
    from safetune.core.eval.pipeline.backends.vllm import VllmBackend

    def fake_llm(*a, **kw):
        raise _hub_error(GatedRepoError, "403 Client Error")

    monkeypatch.setattr(vllm, "LLM", fake_llm)
    backend = VllmBackend(model="meta-llama/Llama-3.2-1B-Instruct")
    with pytest.raises(GatedResourceError) as exc_info:
        backend._ensure_loaded()
    assert exc_info.value.kind == "model"
    assert exc_info.value.repo_id == "meta-llama/Llama-3.2-1B-Instruct"
