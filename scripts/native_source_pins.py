"""Verify local inputs, with a commit-blob fallback for dirty source worktrees."""

import hashlib
import re
import subprocess
from pathlib import Path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _committed_blob_sha256(path, source_commit):
    try:
        root = Path(
            subprocess.check_output(
                ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        ).resolve()
        relative = path.resolve().relative_to(root).as_posix()
        content = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{source_commit}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(content).hexdigest()


def verify_pins(pins, source_commit=None):
    """Verify file hashes, falling back only to a matching immutable Git blob."""
    if source_commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError("source commit must be a full 40-character Git object ID")

    committed_fallbacks = []
    for raw_path, expected in pins.items():
        path = Path(raw_path)
        try:
            actual = _file_sha256(path)
        except OSError:
            actual = None
        if actual == expected:
            continue

        if source_commit and _committed_blob_sha256(path, source_commit) == expected:
            committed_fallbacks.append(
                {"path": str(path), "source_commit": source_commit, "sha256": expected}
            )
            continue
        raise ValueError("changed input/source: " + raw_path)
    return committed_fallbacks
