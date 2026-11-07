"""Audit E11: pareto_frontier / ParetoVisualizer against a brute-force check."""
import itertools
import math
import random

import pytest

from safetune.evaluate.suite.pareto import ParetoPoint, ParetoVisualizer, pareto_frontier


def _brute(points, ms, mc):
    def better(a, b):                       # a dominates b
        sa = (a.safety >= b.safety) if ms else (a.safety <= b.safety)
        ca = (a.capability >= b.capability) if mc else (a.capability <= b.capability)
        strict = (a.safety != b.safety) or (a.capability != b.capability)
        return sa and ca and strict
    return [p for p in points if not any(better(q, p) for q in points if q is not p)]


@pytest.mark.parametrize("ms,mc", list(itertools.product([True, False], repeat=2)))
def test_frontier_matches_brute_force_on_random_points_with_ties(ms, mc):
    rng = random.Random(0)
    for _ in range(50):
        pts = [ParetoPoint(str(i), rng.choice([0.1, 0.2, 0.5, 0.9]), rng.choice([0.1, 0.3, 0.7, 1.0])) for i in range(12)]
        got = pareto_frontier(pts, maximize=(ms, mc))
        assert [id(p) for p in got] == [id(p) for p in _brute(pts, ms, mc)]


def test_duplicates_are_both_on_the_frontier_and_order_is_input_order():
    a, b, c = ParetoPoint("a", 0.5, 0.5), ParetoPoint("b", 0.5, 0.5), ParetoPoint("c", 0.1, 0.1)
    assert [p.label for p in pareto_frontier([c, a, b])] == ["a", "b"]
    assert pareto_frontier([]) == []


def test_a_nan_point_is_rejected_instead_of_always_sitting_on_the_frontier():
    """NaN compares False with everything, so it is never dominated: an unscored method is reported as Pareto-optimal."""
    viz = ParetoVisualizer()
    viz.add("good", 0.9, 0.9)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            viz.add("unscored", bad, 0.1)
        with pytest.raises(ValueError, match="finite"):
            viz.add("unscored", 0.1, bad)
    assert [p.label for p in viz.points] == ["good"]


def test_axes_and_labels_of_the_summary():
    viz = ParetoVisualizer(safety_label="asr", capability_label="acc", safety_maximize=False)
    viz.add("low_asr", 0.1, 0.5)
    viz.add("high_asr", 0.8, 0.5)
    rows = {r["label"]: r for r in viz.summary()}
    assert rows["low_asr"]["on_frontier"] and not rows["high_asr"]["on_frontier"]
    assert rows["low_asr"]["asr"] == 0.1 and rows["low_asr"]["acc"] == 0.5
