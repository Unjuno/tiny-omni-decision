"""Freeze only CC0 iNaturalist observer groups not used by the original probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

MIN_NEW_TRAIN = 700
MIN_NEW_VALIDATION = 200
MIN_PER_CLASS_TRAIN = 25
MIN_PER_CLASS_VALIDATION = 10


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["observer_group_sha256"]),
        str(row["observation_id"]),
        str(row["photo_id"]),
    )


def filter_expansion(
    previous_rows: list[dict[str, Any]],
    candidate_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Exclude all previously exposed observer, observation, and photo identities."""
    previous_observers = {str(row["observer_group_sha256"]) for row in previous_rows}
    previous_observations = {str(row["observation_id"]) for row in previous_rows}
    previous_photos = {str(row["photo_id"]) for row in previous_rows}
    retained: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    excluded = Counter()
    seen: dict[str, set[str]] = {
        "observer_taxon": set(),
        "observation_id": set(),
        "photo_id": set(),
    }

    for split in ("train", "validation"):
        for row in candidate_rows[split]:
            if row.get("split") != split:
                raise ValueError(f"candidate row is stored under the wrong split: {split}")
            if row.get("observation_license") != "cc0" or row.get("photo_license") != "cc0":
                raise ValueError("all retained observations and photos must be explicitly CC0")
            observer, observation, photo = _identity(row)
            if observer in previous_observers:
                excluded["previous_observer"] += 1
                continue
            if observation in previous_observations:
                excluded["previous_observation"] += 1
                continue
            if photo in previous_photos:
                excluded["previous_photo"] += 1
                continue
            observer_taxon = f"{observer}:{row['taxon_id']}"
            if observer_taxon in seen["observer_taxon"]:
                raise ValueError("candidate contains duplicate observer/class groups")
            if observation in seen["observation_id"]:
                raise ValueError("candidate contains duplicate observation IDs")
            if photo in seen["photo_id"]:
                raise ValueError("candidate contains duplicate photo IDs")
            seen["observer_taxon"].add(observer_taxon)
            seen["observation_id"].add(observation)
            seen["photo_id"].add(photo)
            retained[split].append(row)

    for split in retained:
        retained[split].sort(key=lambda row: (int(row["taxon_id"]), str(row["observation_id"])))
    _validate_disjoint(retained)
    per_class = {
        split: dict(
            sorted(
                Counter(str(row["taxon_id"]) for row in rows).items(),
                key=lambda item: int(item[0]),
            )
        )
        for split, rows in retained.items()
    }
    expected_classes = {str(row["taxon_id"]) for row in previous_rows}
    represented_classes = {
        str(row["taxon_id"]) for rows in retained.values() for row in rows
    }
    all_classes_represented = represented_classes == expected_classes
    eligible = (
        len(retained["train"]) >= MIN_NEW_TRAIN
        and len(retained["validation"]) >= MIN_NEW_VALIDATION
        and all_classes_represented
        and all(
            int(per_class["train"].get(class_id, 0)) >= MIN_PER_CLASS_TRAIN
            for class_id in expected_classes
        )
        and all(
            int(per_class["validation"].get(class_id, 0)) >= MIN_PER_CLASS_VALIDATION
            for class_id in expected_classes
        )
    )
    summary = {
        "new_train_count": len(retained["train"]),
        "new_validation_count": len(retained["validation"]),
        "excluded_previous_identity_counts": dict(sorted(excluded.items())),
        "per_class": per_class,
        "previous_observer_count": len(previous_observers),
        "identity_disjoint_from_previous": True,
        "train_validation_identity_disjoint": True,
        "all_classes_represented": all_classes_represented,
        "minimum_count_gate": {
            "minimum_new_train": MIN_NEW_TRAIN,
            "minimum_new_validation": MIN_NEW_VALIDATION,
            "minimum_per_class_train": MIN_PER_CLASS_TRAIN,
            "minimum_per_class_validation": MIN_PER_CLASS_VALIDATION,
            "passed": eligible,
        },
        "eligible_for_training": eligible,
    }
    return retained, summary


def _validate_disjoint(splits: dict[str, list[dict[str, Any]]]) -> None:
    for key in ("observer_group_sha256", "observation_id", "photo_id"):
        train = {str(row[key]) for row in splits["train"]}
        validation = {str(row[key]) for row in splits["validation"]}
        if train & validation:
            raise ValueError(f"train/validation {key} overlap")


def freeze_expansion(previous_dir: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    previous_dir = previous_dir.resolve(strict=True)
    source_dir = source_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite expansion output: {output_dir}")
    previous_paths = [
        previous_dir / "metadata.json",
        previous_dir / "train.jsonl",
        previous_dir / "validation.jsonl",
    ]
    candidate_paths = [
        source_dir / "metadata.json",
        source_dir / "train.jsonl",
        source_dir / "validation.jsonl",
    ]
    previous_rows = [row for path in previous_paths[1:] for row in _read_jsonl(path)]
    candidate = {
        split: _read_jsonl(source_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    retained, summary = filter_expansion(previous_rows, candidate)

    output_dir.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for split, rows in retained.items():
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ).encode("utf-8")
        path = output_dir / f"{split}.jsonl"
        path.write_bytes(payload)
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
    metadata = {
        "schema_version": "inat-cc0-bird-expansion-v1",
        "source": "iNaturalist public API frozen response; underlying API is live",
        "previous_metadata_sha256": {path.name: _sha256(path) for path in previous_paths},
        "expanded_source_sha256": {
            path.name: _sha256(path) for path in candidate_paths
        },
        "split": {
            "inherited_seed": 17,
            "group": "observer SHA-256 from source freeze",
            "sealed_audit_materialized": False,
            "previous_validation_reused": False,
        },
        "privacy": {
            "raw_observer_ids_retained": False,
            "observer_names_retained": False,
            "coordinates_retained": False,
            "descriptions_retained": False,
        },
        "summary": summary,
        "sha256": hashes,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json.sha256").write_text(
        f"{_sha256(metadata_path)}  metadata.json\n", encoding="ascii"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--expanded-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_expansion(args.previous_dir, args.expanded_source_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
