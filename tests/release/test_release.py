import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release import (
    INIT_PATH,
    PR_MARKER,
    PYPROJECT_PATH,
    ReleaseError,
    classify_merged_pr,
    classify_pypi_files,
    decide_bump_state,
    decide_release_state,
    is_open_bump_pr,
    next_patch,
    package_version,
    parse_semver,
    validate_requested_version,
    verify_distributions,
    write_next_patch,
)


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.0.0", "0.0.1"), ("1.2.9", "1.2.10"), ("10.20.99", "10.20.100")],
)
def test_next_patch_only_increments_patch(current: str, expected: str) -> None:
    assert next_patch(current) == expected


@pytest.mark.parametrize(
    "version",
    ["v1.2.3", "1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "1.2.3-rc.1", "1.2.3+build", ""],
)
def test_invalid_plain_semver_is_rejected(version: str) -> None:
    with pytest.raises(ReleaseError):
        parse_semver(version)


def _write_version_files(root: Path, pyproject_version: str, runtime_version: str) -> None:
    (root / INIT_PATH).parent.mkdir(parents=True)
    (root / PYPROJECT_PATH).write_text(
        f'[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "demo"\nversion = "{pyproject_version}"\n',
        encoding="utf-8",
    )
    (root / INIT_PATH).write_text(
        f'"""Demo."""\n\n__version__ = "{runtime_version}"\n', encoding="utf-8"
    )


def test_write_next_patch_updates_both_version_sources(tmp_path: Path) -> None:
    _write_version_files(tmp_path, "2.4.9", "2.4.9")

    assert write_next_patch(tmp_path) == "2.4.10"
    assert package_version(tmp_path) == "2.4.10"


def test_mismatched_tracked_versions_are_rejected(tmp_path: Path) -> None:
    _write_version_files(tmp_path, "2.4.9", "2.4.8")

    with pytest.raises(ReleaseError, match="do not match"):
        package_version(tmp_path)


def test_requested_tag_and_package_must_match() -> None:
    assert validate_requested_version("1.2.3", "1.2.3", "1.2.3") == "1.2.3"
    with pytest.raises(ReleaseError, match="Requested version"):
        validate_requested_version("1.2.3", "1.2.4", "1.2.4")
    with pytest.raises(ReleaseError, match="Tag"):
        validate_requested_version("1.2.3", "1.2.3", "1.2.4")


def test_normal_merge_is_not_treated_as_a_release() -> None:
    assert classify_merged_pr(
        base_version="1.2.3",
        merged_version="1.2.3",
        head_ref="feature/useful-change",
        title="Add a useful change",
        body="",
        head_repository="owner/repo",
        repository="owner/repo",
        files={"src/safetune/feature.py"},
    ) == ("normal", "1.2.3")


def test_valid_generated_bump_merge_is_recognized() -> None:
    assert classify_merged_pr(
        base_version="1.2.3",
        merged_version="1.2.4",
        head_ref="automation/patch-v1.2.4",
        title="chore(release): bump version to 1.2.4",
        body=PR_MARKER,
        head_repository="owner/repo",
        repository="owner/repo",
        files={str(PYPROJECT_PATH), str(INIT_PATH)},
    ) == ("release", "1.2.4")


@pytest.mark.parametrize(
    "changes",
    [
        {"merged_version": "1.3.0"},
        {"title": "chore(release): bump version to 9.9.9"},
        {"body": ""},
        {"head_repository": "fork/repo"},
        {"files": {str(PYPROJECT_PATH), str(INIT_PATH), "README.md"}},
    ],
)
def test_malformed_generated_bump_merge_is_rejected(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "base_version": "1.2.3",
        "merged_version": "1.2.4",
        "head_ref": "automation/patch-v1.2.4",
        "title": "chore(release): bump version to 1.2.4",
        "body": PR_MARKER,
        "head_repository": "owner/repo",
        "repository": "owner/repo",
        "files": {str(PYPROJECT_PATH), str(INIT_PATH)},
    }
    values.update(changes)

    with pytest.raises(ReleaseError):
        classify_merged_pr(**values)  # type: ignore[arg-type]


def test_open_bump_pr_detection_requires_all_markers() -> None:
    pull_request = {
        "headRefName": "automation/patch-v1.2.4",
        "title": "chore(release): bump version to 1.2.4",
        "body": PR_MARKER,
    }
    assert is_open_bump_pr(pull_request)
    assert not is_open_bump_pr({**pull_request, "body": ""})


def test_bump_bootstraps_an_unreleased_current_version() -> None:
    assert decide_bump_state(tag=True, release=True, pypi="complete") == "create"
    assert decide_bump_state(tag=False, release=False, pypi="absent") == "bootstrap"
    assert decide_bump_state(tag=True, release=False, pypi="absent") == "bootstrap"
    assert decide_bump_state(tag=True, release=True, pypi="absent") == "bootstrap"


def test_bump_rejects_an_inconsistent_current_release() -> None:
    with pytest.raises(ReleaseError, match="partially released"):
        decide_bump_state(tag=False, release=True, pypi="absent")
    with pytest.raises(ReleaseError, match="partially released"):
        decide_bump_state(tag=True, release=True, pypi="partial")


@pytest.mark.parametrize(
    ("tag", "release", "pypi", "expected"),
    [
        (False, False, "absent", "create-tag"),
        (True, False, "absent", "create-release"),
        (True, True, "absent", "publish"),
        (True, True, "complete", "complete"),
    ],
)
def test_release_rerun_states_are_safe(tag: bool, release: bool, pypi: str, expected: str) -> None:
    assert decide_release_state(tag=tag, release=release, pypi=pypi) == expected


@pytest.mark.parametrize(
    ("tag", "release", "pypi"),
    [(False, True, "absent"), (False, False, "complete"), (True, True, "partial")],
)
def test_inconsistent_release_states_are_rejected(tag: bool, release: bool, pypi: str) -> None:
    with pytest.raises(ReleaseError, match="Inconsistent release state"):
        decide_release_state(tag=tag, release=release, pypi=pypi)


def test_pypi_version_requires_wheel_and_sdist() -> None:
    assert classify_pypi_files([]) == "absent"
    assert classify_pypi_files(["safetune-1.2.3.tar.gz"]) == "partial"
    assert (
        classify_pypi_files(["safetune-1.2.3.tar.gz", "safetune-1.2.3-py3-none-any.whl"])
        == "complete"
    )


def test_distribution_metadata_must_match_requested_version(tmp_path: Path) -> None:
    wheel = tmp_path / "safetune-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("safetune-1.2.3.dist-info/METADATA", "Version: 1.2.3\n")

    sdist = tmp_path / "safetune-1.2.3.tar.gz"
    metadata = b"Version: 1.2.3\n"
    with tarfile.open(sdist, "w:gz") as archive:
        canonical = tarfile.TarInfo("safetune-1.2.3/PKG-INFO")
        canonical.size = len(metadata)
        archive.addfile(canonical, io.BytesIO(metadata))
        duplicate = tarfile.TarInfo("safetune-1.2.3/src/safetune.egg-info/PKG-INFO")
        duplicate.size = len(metadata)
        archive.addfile(duplicate, io.BytesIO(metadata))

    verify_distributions([wheel, sdist], "1.2.3")
    with pytest.raises(ReleaseError, match="expected '1.2.4'"):
        verify_distributions([wheel, sdist], "1.2.4")
