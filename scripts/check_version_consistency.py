#!/usr/bin/env python3
"""Fail if the declared version is not the truth, or the CHANGELOG overclaims.

SafeTune's release pipeline is also its only CI, and it failed at the test gate
on four consecutive merged PRs (#18-#22) while PyPI served 0.1.5: the pipeline
bumps the tree to the next patch version before running the suite, and the
suite hardcoded the old literal. Both sides looked correct -- the tree agreed
with itself -- they agreed on something that blocked every release.

The same check runs in Lexsi-Labs/AlignTune-Internal and Lexsi-Labs/CuratorKIT,
where the same drift class sat unnoticed across releases.

Three checks:

1. the declared version is valid semver, and CITATION.cff and pyproject.toml
   declare the same one,
2. no dated CHANGELOG heading claims a version higher than the declared one,
3. (opt-in, ``--pypi``) the declared version is not behind PyPI's latest.

Check 2 is the one that matters. Check 3 needs the network, so it is off by
default; run it in CI or before a release, where being behind the index is
worth failing on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
# "## 1.0.0 - 2026-06-12" and "## 0.2.0" both count. "## [Unreleased]" and
# "## [Unreleased] - 2026-07-31" do not, which is the point: an Unreleased heading
# cannot overclaim, because nothing has shipped under it.
HEADING = re.compile(r"^##\s+\[?(\d+\.\d+\.\d*)\]?", re.M)
CITATION = re.compile(r"^version:\s*[\"']?([^\"'\s#]+)", re.M)
PYPROJECT = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.M)


def parse(value: str) -> tuple[int, ...]:
    return tuple(int(p) for p in value.split("."))


def declared_version() -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m = CITATION.search(text)
    if not m:
        raise SystemExit("FAIL: no version field in CITATION.cff")
    return m.group(1)


def pypi_latest(project: str) -> str | None:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/json", timeout=10) as r:
            return json.load(r)["info"]["version"]
    except Exception as exc:  # noqa: BLE001 -- offline is not a failure, it is unknown
        print(f"  (could not reach PyPI: {exc})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pypi", action="store_true", help="also require the declared version to match PyPI")
    ap.add_argument("--project", default="safetune")
    args = ap.parse_args()

    problems: list[str] = []

    declared = declared_version()
    if not SEMVER.match(declared):
        print(f"FAIL: {declared!r} in CITATION.cff is not valid semver.")
        return 1

    pyproject = PYPROJECT.search((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if pyproject and pyproject.group(1) != declared:
        problems.append(
            f"pyproject.toml declares {pyproject.group(1)}, but CITATION.cff declares {declared}."
            "\n  scripts/release.py current_version() refuses to run on this state; keep them equal."
        )

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for h in HEADING.finditer(changelog):
        claimed = h.group(1)
        try:
            if parse(claimed) > parse(declared):
                problems.append(
                    f"CHANGELOG.md claims {claimed}, but the package declares {declared}."
                    "\n  A dated heading for a version that was never published reads as"
                    "\n  shipped. Move it under '## [Unreleased]' or drop the date."
                )
        except ValueError:
            continue

    if args.pypi:
        latest = pypi_latest(args.project)
        if latest and parse(declared) < parse(latest):
            problems.append(
                f"CITATION.cff declares {declared}, but PyPI's latest {args.project} is {latest}."
                "\n  Either the version files are stale, or the release did not happen."
            )

    if problems:
        print(f"FAIL: declared {declared}, but:")
        for p in problems:
            print(f"  - {p}")
        return 1

    extra = ""
    if args.pypi:
        latest = pypi_latest(args.project)
        extra = f", PyPI latest {latest}" if latest else ""
    print(f"OK: declared {declared}{extra}, and no heading overclaims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
