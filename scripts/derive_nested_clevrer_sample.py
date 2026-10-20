"""Create a scene-nested CLEVRER train subset without downloading media again."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.textfree_video import validate_video_splits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    contents = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(contents, encoding="utf-8")


def _scene_order(scene_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}|train|scene|{scene_id}".encode()).hexdigest()


def select_nested_train_rows(
    rows: list[dict[str, Any]], *, scene_count: int, seed: int
) -> list[dict[str, Any]]:
    if scene_count < 1:
        raise ValueError("scene_count must be positive")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("nested train input contains a non-train row")
        grouped[int(row["scene_index"])].append(row)
    if len(grouped) < scene_count:
        raise ValueError(f"need {scene_count} scenes, found {len(grouped)}")
    eligible = []
    for scene, scene_rows in grouped.items():
        kinds = {row.get("question_type") for row in scene_rows}
        if kinds != {"static_descriptive", "temporal_descriptive"}:
            raise ValueError(f"scene {scene} does not have both frozen task types")
        eligible.append(scene)
    selected = set(sorted(eligible, key=lambda scene: _scene_order(scene, seed))[:scene_count])
    result = [row for row in rows if int(row["scene_index"]) in selected]
    return sorted(result, key=lambda row: (int(row["scene_index"]), int(row["question_id"])))


def _materialize_media(
    source_dir: Path, output_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, int]:
    linked = copied = 0
    for relative in sorted({str(row["media_path"]) for row in rows}):
        source = (source_dir / relative).resolve(strict=True)
        try:
            source.relative_to(source_dir)
        except ValueError as error:
            raise ValueError(f"source media path escapes frozen sample: {source}") from error
        target = output_dir / relative
        if not target.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"output media path escapes nested sample: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        try:
            os.link(source, target)
            linked += 1
        except OSError:
            shutil.copy2(source, target)
            copied += 1
        expected_hashes = {
            str(row["media_sha256"]) for row in rows if row["media_path"] == relative
        }
        if len(expected_hashes) != 1 or _sha256(target) != next(iter(expected_hashes)):
            raise ValueError(f"media identity mismatch after materializing {relative}")
    return {"hardlinked_media_files": linked, "copied_media_files": copied}


def derive_sample(
    source_dir: Path, output_dir: Path, *, scene_count: int, seed: int
) -> dict[str, Any]:
    source_dir = source_dir.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite nested sample: {output_dir}")
    train_rows = _read_rows(source_dir / "train.jsonl")
    validation_rows = _read_rows(source_dir / "validation.jsonl")
    selected_train = select_nested_train_rows(
        train_rows, scene_count=scene_count, seed=seed
    )
    if len(selected_train) != 2 * scene_count:
        raise ValueError("expected exactly one row per descriptive question type and scene")
    validate_video_splits(selected_train, validation_rows, allow_normalized_content_overlap=True)

    output_dir.mkdir(parents=True)
    train_path, validation_path = output_dir / "train.jsonl", output_dir / "validation.jsonl"
    _write_rows(train_path, selected_train)
    _write_rows(validation_path, validation_rows)
    media_counts = _materialize_media(
        source_dir, output_dir, selected_train + validation_rows
    )
    source_fetch_path = source_dir / "fetch-report.json"
    source_fetch = json.loads(source_fetch_path.read_text(encoding="utf-8"))
    source_fetch_sha256 = _sha256(source_fetch_path)
    fetch_report = {
        "status": "derived_nested_clevrer_sample",
        "source": source_fetch.get("source", "MIT-IBM/CLEVRER"),
        "revision": source_fetch.get("revision", "98b842082ba4f7c18b6b9e3f39145871782a65ef"),
        "license": "CC0-1.0",
        "seed": seed,
        "train_unique_scenes": scene_count,
        "train_rows": len(selected_train),
        "validation_unique_scenes": len({int(row["scene_index"]) for row in validation_rows}),
        "validation_rows": len(validation_rows),
        "parent_sample_dir": str(source_dir),
        "parent_train_manifest_sha256": _sha256(source_dir / "train.jsonl"),
        "parent_validation_manifest_sha256": _sha256(source_dir / "validation.jsonl"),
        "parent_fetch_report_sha256": source_fetch_sha256,
        **media_counts,
    }
    fetch_path = output_dir / "fetch-report.json"
    fetch_path.write_text(
        json.dumps(fetch_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scene_train = {int(row["scene_index"]) for row in selected_train}
    scene_validation = {int(row["scene_index"]) for row in validation_rows}
    media_train = {str(row["media_sha256"]) for row in selected_train}
    media_validation = {str(row["media_sha256"]) for row in validation_rows}
    if scene_train & scene_validation or media_train & media_validation:
        raise ValueError("scene or media identity overlap after nested sample derivation")
    gate = {
        "status": "passed",
        "all_media_sha256_verified": True,
        "group_integrity": {
            "train_validation_scene_overlap": 0,
            "train_validation_media_overlap": 0,
            "train_unique_scenes": len(scene_train),
            "validation_unique_scenes": len(scene_validation),
        },
        "sample_hashes": {
            "train_jsonl": _sha256(train_path),
            "validation_jsonl": _sha256(validation_path),
        },
        "parent_integrity_basis": (
            "selected media are hashed after hardlink/copy; parent sample is immutable"
        ),
    }
    gate_path = output_dir / "media-integrity-v2.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "status": "PASS_NESTED_SAMPLE_DERIVATION",
        "train_rows": len(selected_train),
        "train_unique_scenes": len(scene_train),
        "train_unique_media": len(media_train),
        "validation_rows": len(validation_rows),
        "validation_unique_scenes": len(scene_validation),
        "validation_unique_media": len(media_validation),
        "scene_overlap": 0,
        "media_identity_overlap": 0,
        "parent_fetch_report_sha256": source_fetch_sha256,
        "train_manifest_sha256": gate["sample_hashes"]["train_jsonl"],
        "validation_manifest_sha256": gate["sample_hashes"]["validation_jsonl"],
        "fetch_report_sha256": _sha256(fetch_path),
        "integrity_report_sha256": _sha256(gate_path),
        **media_counts,
    }
    result_path = output_dir / "nested-sample-report.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    result = derive_sample(
        args.source_dir,
        args.output_dir,
        scene_count=args.train_scenes,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
