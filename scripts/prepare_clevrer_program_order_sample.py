"""Prepare a scene-disjoint, previously unused CLEVRER validation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from requests.exceptions import RequestException

from scripts.fetch_clevrer_probe import (
    REVISION,
    VALIDATION_BYTES,
    VALIDATION_ETAG,
    VALIDATION_QUESTIONS_SHA256,
    VALIDATION_URL,
    RangeFile,
    choose_validation,
    sha256,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_content(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = " ".join(str(row["question"]).casefold().split())
    options = tuple(sorted(" ".join(str(item).casefold().split()) for item in row["options"]))
    return question, options


def discover_previously_observed_scenes(
    artifacts_root: Path,
) -> tuple[set[int], list[dict[str, str]]]:
    """Collect CLEVRER scene IDs from prior JSONL samples and saved predictions."""
    scene_ids: set[int] = set()
    sources: list[dict[str, str]] = []
    relevant = ("clevrer", "textfree-video", "videoprism-clevrer")
    ignored = {".venv", "venv", "feature-env", "node_modules", "site-packages"}
    for top in sorted(artifacts_root.iterdir()):
        if not top.is_dir() or not any(key in top.name.casefold() for key in relevant):
            continue
        for current, directories, names in os.walk(top, onerror=lambda _error: None):
            directories[:] = sorted(name for name in directories if name not in ignored)
            for name in sorted(names):
                if not name.endswith(".jsonl"):
                    continue
                path = Path(current) / name
                found = False
                try:
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            source = str(row.get("source", "")).casefold()
                            sample_id = str(row.get("id", "")).casefold()
                            if "clevrer" not in source and not sample_id.startswith("clevrer-"):
                                continue
                            for value in (
                                row.get("scene_index"),
                                row.get("video_filename"),
                                row.get("scene_group_id"),
                                row.get("id"),
                                row.get("video_id"),
                            ):
                                if isinstance(value, int) and not isinstance(value, bool):
                                    scene_ids.add(value)
                                    found = True
                                elif isinstance(value, str):
                                    matches = re.findall(r"(?:video_)?(\d{5})", value)
                                    if matches:
                                        scene_ids.update(int(match) for match in matches)
                                        found = True
                except (OSError, UnicodeError):
                    continue
                if found:
                    sources.append({"path": str(path), "sha256": _sha256_file(path)})
    return scene_ids, sources


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return "existing"
        raise FileExistsError(f"unexpected pre-existing corpus media: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _read_member_with_retries(archive: zipfile.ZipFile, member: str, attempts: int = 3) -> bytes:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return archive.read(member)
        except RequestException:
            if attempt + 1 == attempts:
                raise
    raise RuntimeError("unreachable archive retry state")


def select_unused_validation(
    train_rows: list[dict[str, Any]],
    raw_validation_rows: list[dict[str, Any]],
    artifacts_root: Path,
    *,
    per_type: int = 45,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, str]]]:
    exclusions, sources = discover_previously_observed_scenes(artifacts_root)
    train_content = {_normalized_content(row) for row in train_rows}
    selected = choose_validation(
        raw_validation_rows,
        per_type,
        seed,
        train_content,
        exclusions,
    )
    selected_scenes = {int(row["scene_index"]) for row in selected}
    if len(selected) != per_type * 2 or len(selected_scenes) != per_type:
        raise ValueError("selected validation rows are not two task types per unique scene")
    if selected_scenes & exclusions:
        raise ValueError("selected validation includes a previously observed scene")
    if train_content & {_normalized_content(row) for row in selected}:
        raise ValueError("normalized question/options overlap with frozen training content")
    return selected, exclusions, sources


def prepare_sample(
    *,
    raw_validation_path: Path,
    training_sample_dir: Path,
    artifacts_root: Path,
    output_dir: Path,
    per_type: int = 45,
    seed: int = 17,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing sample: {output_dir}")
    if per_type < 1:
        raise ValueError("per_type must be positive")
    train_path = training_sample_dir / "train.jsonl"
    raw = raw_validation_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != VALIDATION_QUESTIONS_SHA256:
        raise ValueError("validation question file does not match the pinned CLEVRER revision")
    train_rows = [
        json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line
    ]
    raw_rows = json.loads(raw)
    selected, exclusions, exclusion_sources = select_unused_validation(
        train_rows, raw_rows, artifacts_root, per_type=per_type, seed=seed
    )
    selected_scenes = {int(row["scene_index"]) for row in selected}

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(train_path, output_dir / "train.jsonl")
    train_link_methods: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for row in train_rows:
        source = (training_sample_dir / str(row["media_path"])).resolve(strict=True)
        if _sha256_file(source) != row["media_sha256"]:
            raise ValueError(f"frozen training media hash mismatch: {row['id']}")
        method = _link_or_copy(source, output_dir / str(row["media_path"]))
        train_link_methods[method] += 1

    archive_bytes_read = 0
    extracted_by_scene: dict[int, tuple[str, int, str]] = {}
    archive = RangeFile(VALIDATION_URL, VALIDATION_BYTES, VALIDATION_ETAG)
    with zipfile.ZipFile(archive) as zipped:
        names_by_basename: dict[str, list[str]] = {}
        for member in zipped.namelist():
            if member.casefold().endswith(".mp4"):
                names_by_basename.setdefault(Path(member).name, []).append(member)
        for row in selected:
            scene = int(row["scene_index"])
            if scene in extracted_by_scene:
                member, size, media_hash = extracted_by_scene[scene]
            else:
                matches = names_by_basename.get(str(row["video_filename"]), [])
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one pinned archive member for {row['video_filename']}"
                    )
                member = matches[0]
                data = _read_member_with_retries(zipped, member)
                size = len(data)
                media_hash = sha256(data)
                media_path = output_dir / "media" / "validation" / str(row["video_filename"])
                media_path.parent.mkdir(parents=True, exist_ok=True)
                with media_path.open("xb") as stream:
                    stream.write(data)
                extracted_by_scene[scene] = (member, size, media_hash)
            row.update(
                {
                    "source": f"MIT-IBM/CLEVRER@{REVISION}",
                    "scene_group_id": f"clevrer:validation:{scene}",
                    "media_path": f"media/validation/{row['video_filename']}",
                    "media_bytes": size,
                    "media_sha256": media_hash,
                    "archive_member": member,
                }
            )
        archive_bytes_read = archive.bytes_read
    with (output_dir / "validation.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    train_media = {str(row["media_sha256"]) for row in train_rows}
    validation_media = {str(row["media_sha256"]) for row in selected}
    train_ids = {str(row["id"]) for row in train_rows}
    validation_ids = {str(row["id"]) for row in selected}
    if train_media & validation_media or train_ids & validation_ids:
        raise ValueError("train/validation sample or media identity overlap")
    report = {
        "status": "PASS_CLEVRER_PROGRAM_ORDER_SAMPLE",
        "dataset": "MIT-IBM/CLEVRER",
        "revision": REVISION,
        "license": "CC0-1.0",
        "seed": seed,
        "train_manifest_sha256": _sha256_file(output_dir / "train.jsonl"),
        "validation_manifest_sha256": _sha256_file(output_dir / "validation.jsonl"),
        "train_id_order_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in train_rows).encode()
        ).hexdigest(),
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in selected).encode()
        ).hexdigest(),
        "train_scene_ids": sorted({int(row["scene_index"]) for row in train_rows}),
        "validation_scene_ids": sorted(selected_scenes),
        "training_source_manifest_sha256": _sha256_file(train_path),
        "validation_questions_sha256": hashlib.sha256(raw).hexdigest(),
        "previously_observed_scene_count": len(exclusions),
        "previously_observed_source_files": exclusion_sources,
        "excluded_scene_ids_sha256": hashlib.sha256(
            "\n".join(map(str, sorted(exclusions))).encode()
        ).hexdigest(),
        "train_rows": len(train_rows),
        "validation_rows": len(selected),
        "train_scenes": len({str(row["scene_group_id"]) for row in train_rows}),
        "validation_scenes": len(selected_scenes),
        "train_unique_media": len(train_media),
        "validation_unique_media": len(validation_media),
        "media_sha_overlap": 0,
        "sample_id_overlap": 0,
        "normalized_question_overlap": 0,
        "train_question_type_counts": {
            task: sum(str(row["question_type"]) == task for row in train_rows)
            for task in sorted({str(row["question_type"]) for row in train_rows})
        },
        "validation_question_type_counts": {
            task: sum(str(row["question_type"]) == task for row in selected)
            for task in sorted({str(row["question_type"]) for row in selected})
        },
        "archive": {
            "url": VALIDATION_URL,
            "bytes": VALIDATION_BYTES,
            "etag": VALIDATION_ETAG,
            "range_bytes_read": archive_bytes_read,
            "archive_sha256_verified": False,
        },
        "train_media_link_methods": train_link_methods,
    }
    (output_dir / "fetch-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = {
        "status": "pretraining_gate_pass",
        "revision": REVISION,
        "license": "CC0-1.0",
        "train_rows": len(train_rows),
        "validation_rows": len(selected),
        "train_unique_scenes": report["train_scenes"],
        "validation_unique_scenes": len(selected_scenes),
        "media_sha_overlap": 0,
        "normalized_question_overlap": 0,
        "archive_sha256_verified": False,
        "excluded_scene_manifest_sha256": report["excluded_scene_ids_sha256"],
    }
    (output_dir / "pretraining-gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-validation-questions", type=Path, required=True)
    parser.add_argument("--training-sample-dir", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=45)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    report = prepare_sample(
        raw_validation_path=args.raw_validation_questions.resolve(strict=True),
        training_sample_dir=args.training_sample_dir.resolve(strict=True),
        artifacts_root=args.artifacts_root.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        per_type=args.per_type,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
