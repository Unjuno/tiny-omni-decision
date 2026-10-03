from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .corpus import file_sha256

_SELECTION_FIELDS = {
    "teacher_id",
    "checkpoint_sha256",
    "config_sha256",
    "train_corpus_sha256",
    "validation_corpus_sha256",
    "sealed_audit_sha256",
    "sealed_audit_manifest_sha256",
    "seed",
    "best_step",
    "selection_rule",
    "sampling_policy",
}


def artifact_sha256(path: Path) -> str:
    """Hash a file or a directory tree using sorted relative paths and file hashes."""
    import hashlib

    root = path.resolve()
    if root.is_file():
        return file_sha256(str(root))
    if not root.is_dir():
        raise ValueError(f"selected checkpoint does not exist: {path}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("selected checkpoint directory is empty")
    digest = hashlib.sha256()
    for item in files:
        resolved = item.resolve()
        if root not in resolved.parents:
            raise ValueError("selected checkpoint contains a path outside its directory")
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(str(item))))
    return digest.hexdigest()


def freeze_teacher_selection(lock_path: Path, selection: dict[str, Any]) -> dict[str, Any]:
    """Write the immutable candidate lock that must precede sealed evaluation."""
    missing = _SELECTION_FIELDS - selection.keys()
    if missing:
        raise ValueError(f"selection lock is missing fields: {sorted(missing)}")
    for key in (
        "checkpoint_sha256",
        "config_sha256",
        "train_corpus_sha256",
        "validation_corpus_sha256",
        "sealed_audit_sha256",
        "sealed_audit_manifest_sha256",
    ):
        value = selection[key]
        valid_hex_digest = isinstance(value, str) and len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )
        if not valid_hex_digest:
            raise ValueError(f"selection lock has an invalid {key}")
    if selection["seed"] < 0 or selection["best_step"] < 1:
        raise ValueError("selection lock seed and selected step must be nonnegative/positive")
    payload = {"schema_version": 1, "frozen": True, **selection}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError("candidate selection is already frozen") from exc
    return payload


def claim_sealed_audit_evaluation_once(
    selection_lock_path: Path,
    audit_manifest_path: Path,
    audit_data_path: Path,
    claim_path: Path,
) -> dict[str, Any]:
    """Claim the single audit evaluation after candidate and audit hashes are frozen."""
    if not selection_lock_path.is_file():
        raise ValueError("freeze candidate selection before sealed audit evaluation")
    selection = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    if selection.get("frozen") is not True:
        raise ValueError("candidate selection lock is not frozen")
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(str(audit_manifest_path)) != selection.get(
        "sealed_audit_manifest_sha256"
    ):
        raise ValueError("sealed audit manifest differs from frozen candidate selection")
    audit_hash = audit_manifest.get("sealed_audit_sha256")
    if audit_hash != selection.get("sealed_audit_sha256"):
        raise ValueError("sealed audit hash does not match frozen candidate selection")
    if audit_manifest.get("sealed_audit_path") not in {None, audit_data_path.name}:
        raise ValueError("sealed audit data path does not match its frozen manifest")
    if file_sha256(str(audit_data_path)) != audit_hash:
        raise ValueError("sealed audit data hash does not match its frozen manifest")
    claim = {
        "schema_version": 1,
        "status": "started",
        "teacher_id": selection["teacher_id"],
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "sealed_audit_sha256": audit_hash,
        "sealed_audit_data_path": audit_data_path.name,
        "selection_lock_sha256": file_sha256(str(selection_lock_path)),
        "audit_manifest_sha256": file_sha256(str(audit_manifest_path)),
        "attempted_at_utc": datetime.now(UTC).isoformat(),
    }
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with claim_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(claim, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError("sealed audit evaluation has already been attempted") from exc
    return claim
