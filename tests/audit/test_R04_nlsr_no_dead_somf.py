"""Audit R04 (review of #109): nlsr.py carried an unreferenced copy of somf_merge with the old
ft - base formula. The only somf_merge is the one in merge.py (ft - aligned)."""
import inspect

from safetune.interventions.recover import merge, nlsr


def test_nlsr_has_no_second_somf_merge():
    assert not hasattr(nlsr, "somf_merge")
    assert callable(merge.somf_merge)


def test_nlsr_has_no_dead_merge_copies():
    # recover/__init__.py imports only apply_nlsr from nlsr.py; the task_arithmetic copy added a
    # tied weight's safety vector twice (2.0, expected 1.0). The live ones are in merge.py.
    for name in ("task_arithmetic", "apply_prepost_merge", "_resolve_pre_model"):
        assert not hasattr(nlsr, name), name
    assert "SOMF" not in inspect.getsource(nlsr)
    assert callable(merge.task_arithmetic) and callable(merge.apply_prepost_merge)
