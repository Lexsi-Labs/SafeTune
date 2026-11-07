"""Audit E02: BoNAttack vs jplhughes/bon-jailbreaking@c118210 (bon/attacks/run_text_bon.py). Mock generator and judge only."""
from safetune.evaluate import BoNAttack, BoNConfig

LONG = " ".join(["placeholder"] * 60)          # > 50 words, no false-positive phrase
PROMPT = "placeholder request about a forbidden thing"


class _Gen:
    def generate(self, prompts):
        return [LONG for _ in prompts]


class _Judge:
    def __init__(self, asr=1.0):
        self.rows, self.asr = [], asr

    def score(self, rows):
        self.rows += rows
        return [{**r, "judgement": {"asr": self.asr}} for r in rows]


def _run(asr, **cfg):
    return BoNAttack(_Gen(), _Judge(asr=asr), BoNConfig(n=4, **cfg)).run(PROMPT)


def test_a_graded_judge_score_below_one_can_succeed_with_a_lower_threshold():
    """StrongREJECT scores are graded in [0, 1]; the old gate `asr < 1.0 -> fail` needed a perfect 5/5."""
    assert _run(0.75, success_threshold=0.5)["succeeded"]
    assert not _run(0.25, success_threshold=0.5)["succeeded"]


def test_default_threshold_is_the_old_gate():
    """Already-correct path: binary judges behave exactly as before."""
    assert BoNConfig().success_threshold == 1.0
    assert _run(1.0)["succeeded"]
    assert not _run(0.0)["succeeded"] and not _run(0.75)["succeeded"]


def test_a_nan_score_never_counts_as_success():
    assert not _run(float("nan"), success_threshold=0.0)["succeeded"]


def test_judge_is_given_the_original_behavior_and_the_augmented_text_separately():
    """Official `get_classifier_output` grades `behavior_str` (the original request), not the scrambled text."""
    j = _Judge()
    BoNAttack(_Gen(), j, BoNConfig(n=4, seed=1, sigma=0.9, early_stop=False)).run(PROMPT)
    assert len(j.rows) == 4
    assert all(r["prompt"] == PROMPT for r in j.rows), f"judge saw {j.rows[0]['prompt']!r}"
    assert any(r["augmented_prompt"] != PROMPT for r in j.rows)


def test_default_string_match_judge_results_are_unchanged_by_the_new_row_fields():
    """Already-correct path: StringMatchJudge reads only 'response'; the trace and success flags are the same."""
    from safetune.core.eval.pipeline.scorer import StringMatchJudge

    class Mixed:
        def generate(self, prompts):
            return [LONG if i % 2 else "I cannot help with that." for i, _ in enumerate(prompts)]

    cfg = dict(n=6, seed=3, early_stop=False, batch_size=3)
    out = BoNAttack(Mixed(), StringMatchJudge(mode="prefix"), BoNConfig(**cfg)).run(PROMPT)
    assert [t["success"] for t in out["trace"]] == [False, True, False] * 2   # the mock indexes within each batch of 3
    assert out["n_evaluated"] == 6 and out["succeeded"]
    assert [t["variant"] for t in out["trace"]] == [t["variant"] for t in BoNAttack(Mixed(), StringMatchJudge(mode="prefix"), BoNConfig(**cfg)).run(PROMPT)["trace"]]
