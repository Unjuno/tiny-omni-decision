"""Immutable Git-blob fallback for changed external working-tree files."""

import hashlib
import importlib
import os
import subprocess

import pytest


def _verify_pins(*args, **kwargs):
    try:
        module = importlib.import_module("scripts.native_source_pins")
    except ModuleNotFoundError:
        pytest.fail("commit-pinned source verification is missing")
    verifier = getattr(module, "verify_pins", None)
    assert callable(verifier), "commit-pinned source verification is missing"
    return verifier(*args, **kwargs)


def _git(repo, *args, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env).decode().strip()


def _committed_source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    source = repo / "scripts" / "encoder.py"
    source.parent.mkdir()
    source.write_bytes(b"pinned source bytes\n")
    _git(repo, "add", "scripts/encoder.py")
    env = os.environ.copy()
    env.update(GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.com")
    env.update(GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.com")
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "pin source"],
        check=True,
        capture_output=True,
        env=env,
    )
    commit = _git(repo, "rev-parse", "HEAD")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return repo, source, commit, digest


def test_verifies_committed_blob_when_external_worktree_file_is_dirty(tmp_path):
    repo, source, commit, digest = _committed_source(tmp_path)
    source.write_bytes(b"uncommitted user change\n")

    checked_from_git = _verify_pins({str(source): digest}, source_commit=commit)

    assert checked_from_git == [
        {"path": str(source), "source_commit": commit, "sha256": digest}
    ]
    assert source.read_bytes() == b"uncommitted user change\n"


def test_rejects_a_dirty_file_that_does_not_match_the_pinned_commit(tmp_path):
    _, source, commit, _ = _committed_source(tmp_path)
    source.write_bytes(b"uncommitted user change\n")
    wrong_digest = hashlib.sha256(b"another version\n").hexdigest()

    with pytest.raises(ValueError, match="changed input/source"):
        _verify_pins({str(source): wrong_digest}, source_commit=commit)


def test_without_a_commit_pin_a_changed_file_fails_closed(tmp_path):
    _, source, _, digest = _committed_source(tmp_path)
    source.write_bytes(b"uncommitted user change\n")

    with pytest.raises(ValueError, match="changed input/source"):
        _verify_pins({str(source): digest})
