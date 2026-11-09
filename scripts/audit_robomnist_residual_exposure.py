"""Audit RoboMNIST residual group exposure using project metadata only.

This script never opens media, feature caches, model weights, or any path whose
name contains ``sealed``. It reports metadata appearances, not proof of model
training or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

GROUP_ID = re.compile(r"Ar\d_V(?:high|med|low)_Ac\d+_Rp\d+_UNC\d+")
RECEIPT_NAMES = {
    "config.json",
    "manifest.json",
    "metadata.json",
    "predictions.jsonl",
    "result.json",
    "run.json",
    "summary.json",
    "train.json",
    "validation-predictions.jsonl",
    "validation.json",
}
TEXT_SUFFIXES = {".md", ".txt"}
MAX_RECEIPT_BYTES = 20_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_ids_hash(ids: set[str]) -> str:
    payload = "".join(f"{item}\n" for item in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _collect_group_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("group_id") and isinstance(child, str):
                if GROUP_ID.fullmatch(child):
                    found.add(child)
            else:
                _collect_group_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_group_ids(child, found)


def _read_source_inventory(inventory_dir: Path, source_manifest: Path) -> tuple[set[str], str]:
    paths = sorted(inventory_dir.glob("directory-*.json"))
    if len(paths) != 60:
        raise ValueError(f"expected 60 directory metadata files; found {len(paths)}")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    source = manifest.get("source")
    directory_hashes = source.get("directory_metadata_sha256") if isinstance(source, dict) else None
    if not isinstance(directory_hashes, dict) or set(directory_hashes) != {p.name for p in paths}:
        raise ValueError("source manifest does not pin exactly the 60 directory metadata files")
    config_path = inventory_dir / "config.json"
    if not isinstance(source.get("config_sha256"), str) or source["config_sha256"] != _sha256(
        config_path
    ):
        raise ValueError("source config hash mismatch")
    groups: set[str] = set()
    for path in paths:
        if directory_hashes[path.name] != _sha256(path):
            raise ValueError(f"source directory metadata hash mismatch: {path.name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("groups", []):
            group_id = row.get("group_id") if isinstance(row, dict) else None
            if not isinstance(group_id, str) or not GROUP_ID.fullmatch(group_id):
                raise ValueError(f"invalid group id in {path.name}")
            if group_id in groups:
                raise ValueError(f"duplicate source group id: {group_id}")
            groups.add(group_id)
    if len(groups) != 2135:
        raise ValueError(f"expected 2,135 source groups; found {len(groups)}")
    return groups, _sha256(source_manifest)


def _read_prior_pool(prior_run: Path) -> set[str]:
    summary_path = prior_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for flag in ("payload_opened", "training", "model_inference", "sealed_audit_evaluated"):
        if summary.get(flag) is not False:
            raise ValueError(f"prior run is not metadata-only: {flag} is not false")

    ids: set[str] = set()
    for split in ("train", "validation"):
        path = prior_run / f"{split}.json"
        if summary.get(f"{split}_sha256") != _sha256(path):
            raise ValueError(f"prior {split} manifest hash mismatch")
        _collect_group_ids(json.loads(path.read_text(encoding="utf-8")), ids)
    return ids


def _receipt_files(artifact_root: Path) -> list[Path]:
    files: list[Path] = []
    for top in sorted(artifact_root.iterdir()):
        if not top.is_dir() or "robomnist" not in top.name.lower():
            continue
        if "residual-exposure-audit" in top.name.lower():
            continue
        if any("sealed" in part.lower() for part in top.parts):
            continue
        for path in sorted(top.rglob("*")):
            if not path.is_file() or any("sealed" in part.lower() for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name.lower() not in RECEIPT_NAMES:
                continue
            if path.stat().st_size > MAX_RECEIPT_BYTES:
                continue
            files.append(path)
    return files


def audit_residual_exposure(
    *, inventory_dir: Path, source_manifest: Path, prior_run: Path, artifact_root: Path
) -> dict[str, Any]:
    source_ids, source_manifest_sha256 = _read_source_inventory(inventory_dir, source_manifest)
    prior_ids = _read_prior_pool(prior_run)
    if prior_ids - source_ids:
        raise ValueError("prior pool contains a group absent from the pinned source inventory")
    residual = source_ids - prior_ids

    exposures: dict[str, set[str]] = {}
    receipt_rows: list[dict[str, Any]] = []
    scanned = 0
    for path in _receipt_files(artifact_root):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise ValueError(f"could not read metadata receipt {path}") from exc
        scanned += 1
        found = set(GROUP_ID.findall(content)) & residual
        if found:
            relative = str(path)
            exposures[relative] = found
            receipt_rows.append(
                {
                    "path": relative,
                    "sha256": _sha256(path),
                    "matched_group_count": len(found),
                    "matched_group_ids_sha256": _canonical_ids_hash(found),
                }
            )

    unmatched = residual - set().union(*exposures.values()) if exposures else residual
    condition_counts = Counter(group.split("_Rp", maxsplit=1)[0] for group in unmatched)
    pair_units = sum(count // 2 for count in condition_counts.values())
    return {
        "schema_version": 1,
        "scope": "metadata_only_no_media_features_weights_or_sealed_paths",
        "source_manifest_sha256": source_manifest_sha256,
        "source_group_count": len(source_ids),
        "source_group_ids_sha256": _canonical_ids_hash(source_ids),
        "prior_nonsealed_group_count": len(prior_ids),
        "prior_nonsealed_group_ids_sha256": _canonical_ids_hash(prior_ids),
        "residual_group_count": len(residual),
        "residual_group_ids_sha256": _canonical_ids_hash(residual),
        "residual_groups_appearing_in_nonsealed_records": len(residual - unmatched),
        "residual_groups_not_found_in_scanned_records": len(unmatched),
        "scanned_record_count": scanned,
        "metadata_exposure_records": receipt_rows,
        "unmatched_condition_counts": dict(sorted(condition_counts.items())),
        "maximum_pair_units_from_unmatched_groups": pair_units,
        "maximum_rows_at_four_per_pair": pair_units * 4,
        "interpretation": (
            "Record matches prove metadata appearance only. Missing records do not prove that "
            "a group is project-wide unseen. Prior attempt manifests without run summaries "
            "do not establish whether payload, training, or inference occurred."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    result = audit_residual_exposure(
        inventory_dir=args.inventory_dir.resolve(),
        source_manifest=args.source_manifest.resolve(),
        prior_run=args.prior_run.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key != "metadata_exposure_records"}
    print(json.dumps(compact))


if __name__ == "__main__":
    main()
