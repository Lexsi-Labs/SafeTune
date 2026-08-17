#!/usr/bin/env python3
"""Safe, testable helpers for the automated patch-release workflow."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PYPROJECT_PATH = Path("pyproject.toml")
INIT_PATH = Path("src/safetune/__init__.py")
VERSION_FILES = frozenset({str(PYPROJECT_PATH), str(INIT_PATH)})
PR_MARKER = "<!-- safetune-automated-version-bump -->"
BRANCH_RE = re.compile(r"automation/patch-v(?P<version>.+)\Z")
SEMVER_RE = re.compile(r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)\Z")
INIT_VERSION_RE = re.compile(r"^__version__\s*=\s*([\"'])(?P<version>[^\"']+)\1\s*$", re.MULTILINE)


class ReleaseError(ValueError):
    """Raised when release metadata or state violates an invariant."""


@dataclass(frozen=True)
class PackageVersions:
    pyproject: str
    runtime: str

    @property
    def value(self) -> str:
        if self.pyproject != self.runtime:
            raise ReleaseError(
                "Package versions do not match: "
                f"pyproject.toml={self.pyproject!r}, safetune.__version__={self.runtime!r}"
            )
        parse_semver(self.pyproject)
        return self.pyproject


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a plain MAJOR.MINOR.PATCH version, rejecting all extensions."""
    match = SEMVER_RE.fullmatch(version)
    if match is None:
        raise ReleaseError(
            f"Invalid plain SemVer {version!r}; expected MAJOR.MINOR.PATCH "
            "with no prefix, prerelease, build metadata, or leading zeroes"
        )
    return tuple(int(match.group(name)) for name in ("major", "minor", "patch"))


def next_patch(version: str) -> str:
    major, minor, patch = parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def _pyproject_version(text: str) -> str:
    in_project = False
    found: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = re.fullmatch(r'version\s*=\s*"([^"]+)"\s*', stripped)
            if match:
                found.append(match.group(1))
    if len(found) != 1:
        raise ReleaseError(
            f"Expected exactly one [project] version in pyproject.toml, found {len(found)}"
        )
    return found[0]


def _runtime_version(text: str) -> str:
    found = [match.group("version") for match in INIT_VERSION_RE.finditer(text)]
    if len(found) != 1:
        raise ReleaseError(
            f"Expected exactly one __version__ assignment in {INIT_PATH}, found {len(found)}"
        )
    return found[0]


def versions_from_text(pyproject_text: str, init_text: str) -> PackageVersions:
    return PackageVersions(
        pyproject=_pyproject_version(pyproject_text), runtime=_runtime_version(init_text)
    )


def package_version(root: Path = Path(".")) -> str:
    versions = versions_from_text(
        (root / PYPROJECT_PATH).read_text(encoding="utf-8"),
        (root / INIT_PATH).read_text(encoding="utf-8"),
    )
    return versions.value


def validate_requested_version(package: str, requested: str, tag: str | None = None) -> str:
    parse_semver(package)
    parse_semver(requested)
    if package != requested:
        raise ReleaseError(
            f"Requested version {requested!r} does not match package version {package!r}"
        )
    if tag is not None:
        parse_semver(tag)
        if tag != package:
            raise ReleaseError(f"Tag {tag!r} does not match package version {package!r}")
    return package


def _replace_pyproject_version(text: str, old: str, new: str) -> str:
    in_project = False
    replaced = 0
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and re.fullmatch(r'version\s*=\s*"[^"]+"\s*', stripped):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'version = "{new}"{newline}'
            replaced += 1
    if replaced != 1 or _pyproject_version(text) != old:
        raise ReleaseError("Could not safely replace the [project] version in pyproject.toml")
    return "".join(lines)


def _replace_runtime_version(text: str, old: str, new: str) -> str:
    matches = list(INIT_VERSION_RE.finditer(text))
    if len(matches) != 1 or matches[0].group("version") != old:
        raise ReleaseError(f"Could not safely replace __version__ in {INIT_PATH}")
    start, end = matches[0].span("version")
    return f"{text[:start]}{new}{text[end:]}"


def write_next_patch(root: Path = Path(".")) -> str:
    pyproject = root / PYPROJECT_PATH
    runtime = root / INIT_PATH
    pyproject_text = pyproject.read_text(encoding="utf-8")
    runtime_text = runtime.read_text(encoding="utf-8")
    current = versions_from_text(pyproject_text, runtime_text).value
    target = next_patch(current)
    pyproject.write_text(
        _replace_pyproject_version(pyproject_text, current, target), encoding="utf-8"
    )
    runtime.write_text(_replace_runtime_version(runtime_text, current, target), encoding="utf-8")
    if package_version(root) != target:
        raise ReleaseError("Version files did not contain the requested version after writing")
    return target


def _git_text(ref: str, path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def version_at_ref(ref: str) -> str:
    return versions_from_text(_git_text(ref, PYPROJECT_PATH), _git_text(ref, INIT_PATH)).value


def changed_files(base_ref: str, target_ref: str) -> frozenset[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref, target_ref],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return frozenset(line for line in result.stdout.splitlines() if line)


def verify_bump_ref(base_ref: str, bump_ref: str, expected: str) -> str:
    base = version_at_ref(base_ref)
    bumped = version_at_ref(bump_ref)
    validate_requested_version(bumped, expected)
    if next_patch(base) != expected:
        raise ReleaseError(
            f"Existing branch version {expected!r} is not the next patch after {base!r}"
        )
    files = changed_files(base_ref, bump_ref)
    if files != VERSION_FILES:
        raise ReleaseError(
            "Existing automation branch differs from its base outside version files: "
            f"{sorted(files)!r}"
        )
    return expected


def classify_merged_pr(
    *,
    base_version: str,
    merged_version: str,
    head_ref: str,
    title: str,
    body: str,
    head_repository: str,
    repository: str,
    files: Iterable[str],
) -> tuple[str, str]:
    """Return (kind, version), rejecting malformed automated release PRs."""
    branch_match = BRANCH_RE.fullmatch(head_ref)
    has_release_intent = bool(
        branch_match or title.startswith("chore(release): bump version to ") or PR_MARKER in body
    )
    if not has_release_intent:
        return "normal", merged_version

    if branch_match is None:
        raise ReleaseError("A release-like PR used an invalid automation branch name")
    target = branch_match.group("version")
    parse_semver(target)
    expected_title = f"chore(release): bump version to {target}"
    if title != expected_title or PR_MARKER not in body:
        raise ReleaseError("Automated release PR title/body marker validation failed")
    if head_repository != repository:
        raise ReleaseError("Automated release PR must originate in the same repository")
    if frozenset(files) != VERSION_FILES:
        raise ReleaseError(
            f"Automated release PR may change only version files; got {sorted(set(files))!r}"
        )
    expected = next_patch(base_version)
    if target != expected or merged_version != target:
        raise ReleaseError(
            f"Merged release version must be the next patch {expected!r}; "
            f"branch={target!r}, package={merged_version!r}"
        )
    return "release", target


def is_open_bump_pr(pull_request: dict[str, object]) -> bool:
    head_ref = str(pull_request.get("headRefName", ""))
    title = str(pull_request.get("title", ""))
    body = str(pull_request.get("body", ""))
    return bool(
        BRANCH_RE.fullmatch(head_ref)
        and title.startswith("chore(release): bump version to ")
        and PR_MARKER in body
    )


def decide_bump_state(*, tag: bool, release: bool, pypi: str) -> str:
    """Decide whether the current version is complete enough to bump."""
    if pypi not in {"absent", "complete", "partial"}:
        raise ReleaseError(f"Unknown PyPI state {pypi!r}")
    if tag and release and pypi == "complete":
        return "create"
    if not tag and not release and pypi == "absent":
        return "skip"
    raise ReleaseError(
        "Current version is only partially released: "
        f"tag={tag}, github_release={release}, pypi={pypi}"
    )


def decide_release_state(*, tag: bool, release: bool, pypi: str) -> str:
    """Return the safe recovery point for an exact-version release."""
    states = {
        (False, False, "absent"): "create-tag",
        (True, False, "absent"): "create-release",
        (True, True, "absent"): "publish",
        (True, True, "complete"): "complete",
    }
    try:
        return states[(tag, release, pypi)]
    except KeyError as error:
        raise ReleaseError(
            "Inconsistent release state; refusing a duplicate or out-of-order mutation: "
            f"tag={tag}, github_release={release}, pypi={pypi}"
        ) from error


def classify_pypi_files(filenames: Iterable[str]) -> str:
    names = list(filenames)
    if not names:
        return "absent"
    has_wheel = any(name.endswith(".whl") for name in names)
    has_sdist = any(name.endswith(".tar.gz") for name in names)
    return "complete" if has_wheel and has_sdist else "partial"


def pypi_state(project: str, version: str) -> str:
    parse_semver(version)
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return "absent"
        raise ReleaseError(f"PyPI returned HTTP {error.code} for {project} {version}") from error
    filenames = [item["filename"] for item in payload.get("urls", [])]
    return classify_pypi_files(filenames)


def _metadata_version(message: str) -> str:
    versions = [
        line.removeprefix("Version: ").strip()
        for line in message.splitlines()
        if line.startswith("Version: ")
    ]
    if len(versions) != 1:
        raise ReleaseError(
            f"Expected one Version field in distribution metadata, found {len(versions)}"
        )
    return versions[0]


def distribution_version(path: Path) -> str:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ReleaseError(f"Expected one METADATA file in {path}, found {len(metadata)}")
            return _metadata_version(archive.read(metadata[0]).decode("utf-8"))
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/PKG-INFO") and member.name.count("/") == 1
            ]
            if len(members) != 1:
                raise ReleaseError(f"Expected one PKG-INFO file in {path}, found {len(members)}")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise ReleaseError(f"Could not read PKG-INFO from {path}")
            return _metadata_version(extracted.read().decode("utf-8"))
    raise ReleaseError(f"Unsupported distribution file {path}")


def verify_distributions(paths: Sequence[Path], expected: str) -> None:
    parse_semver(expected)
    if not paths:
        raise ReleaseError("No distributions were provided")
    state = classify_pypi_files(path.name for path in paths)
    if state != "complete":
        raise ReleaseError("Expected both a wheel and source distribution")
    for path in paths:
        actual = distribution_version(path)
        if actual != expected:
            raise ReleaseError(f"Distribution {path} has version {actual!r}, expected {expected!r}")


def _bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ReleaseError(f"Expected true or false, got {value!r}")
    return value == "true"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("current")
    subparsers.add_parser("next")
    subparsers.add_parser("bump")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--expected", required=True)
    verify.add_argument("--tag")

    classify = subparsers.add_parser("classify-merge")
    classify.add_argument("--base-ref", required=True)
    classify.add_argument("--merge-ref", required=True)
    classify.add_argument("--head-ref", required=True)
    classify.add_argument("--title", required=True)
    classify.add_argument("--body", default="")
    classify.add_argument("--head-repository", required=True)
    classify.add_argument("--repository", required=True)

    open_bump = subparsers.add_parser("find-open-bump")
    open_bump.add_argument("--file", type=Path, required=True)

    bump_state = subparsers.add_parser("bump-state")
    bump_state.add_argument("--tag", required=True)
    bump_state.add_argument("--release", required=True)
    bump_state.add_argument("--pypi", required=True)

    release_state_parser = subparsers.add_parser("release-state")
    release_state_parser.add_argument("--tag", required=True)
    release_state_parser.add_argument("--release", required=True)
    release_state_parser.add_argument("--pypi", required=True)

    pypi = subparsers.add_parser("pypi-state")
    pypi.add_argument("--project", required=True)
    pypi.add_argument("--version", required=True)

    verify_ref = subparsers.add_parser("verify-ref")
    verify_ref.add_argument("--ref", required=True)
    verify_ref.add_argument("--expected", required=True)

    verify_bump = subparsers.add_parser("verify-bump-ref")
    verify_bump.add_argument("--base-ref", required=True)
    verify_bump.add_argument("--bump-ref", required=True)
    verify_bump.add_argument("--expected", required=True)

    verify_dist = subparsers.add_parser("verify-dist")
    verify_dist.add_argument("--expected", required=True)
    verify_dist.add_argument("paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "current":
            print(package_version())
        elif args.command == "next":
            print(next_patch(package_version()))
        elif args.command == "bump":
            print(write_next_patch())
        elif args.command == "verify":
            print(validate_requested_version(package_version(), args.expected, args.tag))
        elif args.command == "classify-merge":
            kind, version = classify_merged_pr(
                base_version=version_at_ref(args.base_ref),
                merged_version=version_at_ref(args.merge_ref),
                head_ref=args.head_ref,
                title=args.title,
                body=args.body,
                head_repository=args.head_repository,
                repository=args.repository,
                files=changed_files(args.base_ref, args.merge_ref),
            )
            print(f"kind={kind}")
            print(f"version={version}")
        elif args.command == "find-open-bump":
            pull_requests = json.loads(args.file.read_text(encoding="utf-8"))
            matches = [pr for pr in pull_requests if is_open_bump_pr(pr)]
            print(matches[0].get("number", "") if matches else "")
        elif args.command == "bump-state":
            print(
                decide_bump_state(tag=_bool(args.tag), release=_bool(args.release), pypi=args.pypi)
            )
        elif args.command == "release-state":
            print(
                decide_release_state(
                    tag=_bool(args.tag), release=_bool(args.release), pypi=args.pypi
                )
            )
        elif args.command == "pypi-state":
            print(pypi_state(args.project, args.version))
        elif args.command == "verify-ref":
            print(validate_requested_version(version_at_ref(args.ref), args.expected, args.ref))
        elif args.command == "verify-bump-ref":
            print(verify_bump_ref(args.base_ref, args.bump_ref, args.expected))
        elif args.command == "verify-dist":
            verify_distributions(args.paths, args.expected)
            print(args.expected)
        else:  # pragma: no cover - argparse enforces the command choices
            raise ReleaseError(f"Unknown command {args.command!r}")
    except (ReleaseError, subprocess.CalledProcessError) as error:
        print(f"release error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
