"""
Validation matrix script — runs defense × attack grid and writes a CSV.

CSV output columns (schema contract used by downstream tools):
    defense, attack, n, asr, refusal_rate, wall_s
"""
from __future__ import annotations

import csv
import io
import time
from typing import Dict, List


COLUMNS = ["defense", "attack", "n", "asr", "refusal_rate", "wall_s"]


def run_matrix(
    defenses: List[str],
    attacks: List[str],
    prompts: List[str],
    out_path: str | None = None,
) -> List[Dict]:
    """
    Dry-reference orchestrator for the defense × attack validation matrix.

    For each (defense, attack) pair, scores each prompt and records:
        defense, attack, n, asr, refusal_rate, wall_s

    Args:
        defenses: list of defense method identifiers.
        attacks: list of attack/stressor identifiers.
        prompts: list of prompt strings to evaluate.
        out_path: if given, write rows to this CSV path.

    Returns:
        List of row dicts with keys matching COLUMNS.
    """
    rows: List[Dict] = []
    for defense in defenses:
        for attack in attacks:
            t0 = time.monotonic()
            n = len(prompts)
            # Placeholder — real implementation hooks into eval pipeline.
            asr = 0.0
            refusal_rate = 1.0
            wall_s = time.monotonic() - t0
            rows.append(
                {
                    "defense": defense,
                    "attack": attack,
                    "n": n,
                    "asr": asr,
                    "refusal_rate": refusal_rate,
                    "wall_s": wall_s,
                }
            )

    if out_path is not None:
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main():  # pragma: no cover
    import argparse, json

    parser = argparse.ArgumentParser(description="Run defense × attack validation matrix.")
    parser.add_argument("--defenses", nargs="+", required=True)
    parser.add_argument("--attacks", nargs="+", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--out", default="matrix.csv")
    args = parser.parse_args()

    with open(args.prompts_file) as fh:
        prompts = [json.loads(line)["prompt"] for line in fh if line.strip()]

    run_matrix(args.defenses, args.attacks, prompts, out_path=args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
