"""
Pareto-Frontier Visualizer for the safety-vs-capability trade-off.

What it does: collect (safety_score, capability_score) pairs for every
(method, config) tuple in an experiment, identify the Pareto frontier
(no other point dominates), and plot the result.

Domination rule: point A dominates point B iff A.safety >= B.safety AND
A.capability >= B.capability AND at least one inequality is strict.

Why it matters: most defense papers cherry-pick whichever axis flatters
them. A Pareto plot forces the trade-off to be explicit, which is the
single most useful figure a safety paper can include.

The class works without matplotlib for the data-only methods (frontier,
table). matplotlib is required only for ``plot()``; we guard the import
so the rest of the library installs cleanly without it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """One (method, safety, capability) triple, plus optional metadata."""

    label: str
    safety: float
    capability: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def pareto_frontier(points: Iterable[ParetoPoint], maximize: Tuple[bool, bool] = (True, True)) -> List[ParetoPoint]:
    """Compute the Pareto frontier.

    Args:
        points: iterable of :class:`ParetoPoint`.
        maximize: ``(safety_dir, capability_dir)``. Default ``(True, True)``
            means both axes are "more is better". Pass ``False`` for an
            axis where lower is better (e.g. ``ASR``).

    Returns:
        The subset of ``points`` that are not strictly dominated.
    """
    pts = list(points)
    out: List[ParetoPoint] = []
    s_sign = 1.0 if maximize[0] else -1.0
    c_sign = 1.0 if maximize[1] else -1.0

    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            qs, ps = s_sign * q.safety, s_sign * p.safety
            qc, pc = c_sign * q.capability, c_sign * p.capability
            if qs >= ps and qc >= pc and (qs > ps or qc > pc):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return out


class ParetoVisualizer:
    """Collect, summarize, and plot safety-vs-capability points.

    Example::

        viz = ParetoVisualizer(safety_label="refusal_rate", capability_label="gsm8k_acc")
        viz.add("baseline",  safety=0.85, capability=0.30)
        viz.add("ablation",  safety=0.45, capability=0.30)
        viz.add("steer",     safety=0.92, capability=0.27)
        viz.add("probe",     safety=0.88, capability=0.30)
        print(viz.summary())     # table of points with frontier flags
        viz.plot("pareto.png")   # matplotlib scatter; optional
    """

    def __init__(
        self,
        safety_label: str = "safety",
        capability_label: str = "capability",
        safety_maximize: bool = True,
        capability_maximize: bool = True,
    ) -> None:
        self.safety_label = safety_label
        self.capability_label = capability_label
        self.safety_maximize = safety_maximize
        self.capability_maximize = capability_maximize
        self.points: List[ParetoPoint] = []

    def add(self, label: str, safety: float, capability: float, **metadata: Any) -> None:
        self.points.append(
            ParetoPoint(label=label, safety=float(safety), capability=float(capability), metadata=dict(metadata))
        )

    def extend(self, items: Iterable[Tuple[str, float, float]]) -> None:
        for label, s, c in items:
            self.add(label, s, c)

    def frontier(self) -> List[ParetoPoint]:
        return pareto_frontier(self.points, maximize=(self.safety_maximize, self.capability_maximize))

    def summary(self) -> List[Dict[str, Any]]:
        """One row per point with a ``"on_frontier"`` flag."""
        frontier_ids = {id(p) for p in self.frontier()}
        return [
            {
                "label": p.label,
                self.safety_label: p.safety,
                self.capability_label: p.capability,
                "on_frontier": id(p) in frontier_ids,
                **p.metadata,
            }
            for p in self.points
        ]

    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "safety_label": self.safety_label,
            "capability_label": self.capability_label,
            "safety_maximize": self.safety_maximize,
            "capability_maximize": self.capability_maximize,
            "points": self.summary(),
        }
        text = json.dumps(data, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    def plot(self, path: str, title: str = "Safety vs capability", annotate: bool = True) -> None:
        """Render a scatter plot with the Pareto frontier highlighted.

        Requires matplotlib. Raises ImportError if not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover - optional viz dep
            raise ImportError(
                "ParetoVisualizer.plot requires matplotlib. "
                "Install with `pip install matplotlib`."
            ) from e

        if not self.points:
            raise ValueError("ParetoVisualizer.plot: no points to plot.")
        front_ids = {id(p) for p in self.frontier()}

        xs = [p.capability for p in self.points]
        ys = [p.safety for p in self.points]
        on_frontier = [id(p) in front_ids for p in self.points]

        fig, ax = plt.subplots(figsize=(7, 5))
        # Non-frontier points
        ax.scatter(
            [x for x, f in zip(xs, on_frontier) if not f],
            [y for y, f in zip(ys, on_frontier) if not f],
            c="#888888", s=60, label="dominated", zorder=2,
        )
        # Frontier points
        ax.scatter(
            [x for x, f in zip(xs, on_frontier) if f],
            [y for y, f in zip(ys, on_frontier) if f],
            c="#cc4444", s=80, marker="D", label="frontier", zorder=3,
        )
        # Connect frontier in capability-sorted order
        frontier_sorted = sorted(self.frontier(), key=lambda p: p.capability)
        if len(frontier_sorted) >= 2:
            ax.plot(
                [p.capability for p in frontier_sorted],
                [p.safety for p in frontier_sorted],
                "--", c="#cc4444", alpha=0.5, zorder=1,
            )
        if annotate:
            for p in self.points:
                ax.annotate(p.label, (p.capability, p.safety),
                            xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_xlabel(self.capability_label)
        ax.set_ylabel(self.safety_label)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("ParetoVisualizer: wrote %s", out)


__all__ = ["ParetoPoint", "ParetoVisualizer", "pareto_frontier"]
