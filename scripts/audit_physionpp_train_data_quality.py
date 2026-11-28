"""Audit Physion++ TRAIN labels/media and create a stratified visual review set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

VISUAL_FRAMES = 32
THUMBNAIL_SIZE = (128, 72)
GRID_COLUMNS = 8
_TARGET_OBJECT = re.compile(r"(?:^|[-_])target=([^/#-]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_media_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe Physion++ media path: {value}")
    return path


def _scenario_and_target_object(source_record_id: str) -> tuple[str, str]:
    parts = source_record_id.split("/")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"invalid Physion++ source record ID: {source_record_id}")
    match = _TARGET_OBJECT.search(source_record_id)
    target_object = match.group(1) if match else "UNSPECIFIED"
    return parts[1], target_object


def load_train_rows(train_jsonl: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Physion++ train manifest is empty")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("train-only audit refuses non-train rows")
    return rows


def audit_train_rows(rows: list[dict[str, Any]], corpus_root: Path) -> dict[str, Any]:
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("audit accepts only non-empty train rows")

    ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    asset_labels: dict[str, set[str]] = defaultdict(set)
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_object_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_scenarios: dict[str, set[str]] = defaultdict(set)
    frame_counts: list[int] = []
    media_bytes = 0
    label_mismatches = 0

    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError("train sample IDs must be unique non-empty strings")
        ids.add(row_id)
        if row.get("modality") != "video":
            raise ValueError(f"unexpected non-video row: {row_id}")

        source_record_id = row.get("source_record_id")
        if not isinstance(source_record_id, str):
            raise ValueError(f"missing Physion++ source record identity: {row_id}")
        scenario, target_object = _scenario_and_target_object(source_record_id)
        target = row.get("target")
        if target not in {"Yes", "No"}:
            raise ValueError(f"unexpected contact target: {row_id}")
        source_target = row.get("source_target")
        if not isinstance(source_target, bool) or (target == "Yes") != source_target:
            label_mismatches += 1

        media = row.get("media")
        if not isinstance(media, list) or len(media) != 1 or media[0].get("kind") != "video":
            raise ValueError(f"expected one video asset: {row_id}")
        media_path = _relative_media_path(media[0].get("path", ""))
        path_key = media_path.as_posix()
        if path_key in paths:
            raise ValueError(f"duplicate train media path: {path_key}")
        paths.add(path_key)
        expected_sha256 = media[0].get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"missing media SHA-256: {row_id}")
        if expected_sha256 in hashes:
            raise ValueError(f"duplicate train media identity: {expected_sha256}")
        hashes.add(expected_sha256)

        media_file = corpus_root.joinpath(*media_path.parts)
        if not media_file.is_file():
            raise FileNotFoundError(f"missing train media file: {media_file}")
        actual_sha256 = sha256_file(media_file)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"train media hash mismatch: {row_id}")
        media_bytes += media_file.stat().st_size

        frames = media[0].get("num_frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
            raise ValueError(f"invalid frame count: {row_id}")
        frame_counts.append(frames)

        asset_group = row.get("source_asset_group_id")
        if not isinstance(asset_group, str) or not asset_group:
            raise ValueError(f"missing source asset group: {row_id}")
        asset_labels[asset_group].add(target)
        group_scenarios[asset_group].add(scenario)
        scenario_counts[scenario][target] += 1
        target_object_counts[f"{scenario}/{target_object}"][target] += 1

    if label_mismatches:
        raise ValueError(f"{label_mismatches} targets disagree with source_target")
    if any(len(scenarios) != 1 for scenarios in group_scenarios.values()):
        raise ValueError("a source asset group spans multiple Physion++ scenarios")

    ordered_frames = sorted(frame_counts)
    middle = len(ordered_frames) // 2
    median_frames = (
        ordered_frames[middle]
        if len(ordered_frames) % 2
        else (ordered_frames[middle - 1] + ordered_frames[middle]) / 2
    )
    target_counts = Counter(row["target"] for row in rows)
    return {
        "train_rows": len(rows),
        "unique_sample_ids": len(ids),
        "unique_video_paths": len(paths),
        "unique_video_sha256": len(hashes),
        "unique_source_asset_groups": len(asset_labels),
        "asset_groups_with_both_labels": sum(
            len(labels) > 1 for labels in asset_labels.values()
        ),
        "target_counts": dict(sorted(target_counts.items())),
        "source_target_mismatches": 0,
        "scenario_class_counts": {
            scenario: dict(sorted(counts.items()))
            for scenario, counts in sorted(scenario_counts.items())
        },
        "scenario_target_object_class_counts": {
            stratum: dict(sorted(counts.items()))
            for stratum, counts in sorted(target_object_counts.items())
        },
        "matched_scenario_target_object_strata": sum(
            {"No", "Yes"}.issubset(counts) for counts in target_object_counts.values()
        ),
        "scenario_count": len(scenario_counts),
        "frame_count_min": min(frame_counts),
        "frame_count_median": median_frames,
        "frame_count_max": max(frame_counts),
        "verified_media_bytes": media_bytes,
    }


def select_visual_review_rows(
    rows: list[dict[str, Any]], *, per_class_per_scenario: int = 1
) -> list[dict[str, Any]]:
    if per_class_per_scenario < 1:
        raise ValueError("per_class_per_scenario must be positive")
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("visual selection accepts only train rows")

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_record_id = row.get("source_record_id")
        target = row.get("target")
        if not isinstance(source_record_id, str) or target not in {"Yes", "No"}:
            raise ValueError("visual sample rows need source IDs and binary targets")
        scenario, target_object = _scenario_and_target_object(source_record_id)
        buckets[(scenario, target_object, target)].append(row)

    selected: list[dict[str, Any]] = []
    strata = sorted({(scenario, obj) for scenario, obj, _target in buckets})
    for scenario, target_object in strata:
        labels = {
            target
            for target in ("No", "Yes")
            if buckets.get((scenario, target_object, target))
        }
        if labels != {"No", "Yes"}:
            continue
        for target in ("No", "Yes"):
            candidates = buckets[(scenario, target_object, target)]
            ranked = sorted(
                candidates,
                key=lambda row: hashlib.sha256(row["source_record_id"].encode()).hexdigest(),
            )
            selected.extend(ranked[:per_class_per_scenario])
    return selected


def create_contact_sheets(
    rows: list[dict[str, Any]], corpus_root: Path, output_dir: Path
) -> dict[str, Any]:
    try:
        import av
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("contact-sheet generation requires PyAV and Pillow") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_audit: list[dict[str, Any]] = []
    for row in rows:
        scenario, target_object = _scenario_and_target_object(row["source_record_id"])
        row = dict(row)
        row["_visual_target_object"] = target_object
        by_scenario[scenario].append(row)

    for scenario, class_rows in sorted(by_scenario.items()):
        panels: list[tuple[str, dict[str, Any], list[Any], int]] = []
        for row in sorted(class_rows, key=lambda item: (item["target"], item["id"])):
            target = row["target"]
            media = row["media"][0]
            media_path = corpus_root.joinpath(*_relative_media_path(media["path"]).parts)
            expected_frames = media["num_frames"]
            indices = {
                round(i * (expected_frames - 1) / (VISUAL_FRAMES - 1))
                for i in range(VISUAL_FRAMES)
            }
            selected_frames: list[Any] = []
            decoded_count = 0
            with av.open(str(media_path)) as container:
                stream = container.streams.video[0]
                for frame_index, frame in enumerate(container.decode(stream)):
                    decoded_count = frame_index + 1
                    if frame_index in indices:
                        thumbnail = frame.to_image().convert("RGB").resize(THUMBNAIL_SIZE)
                        selected_frames.append((frame_index, thumbnail))
            panels.append((target, row, selected_frames, decoded_count))
            frame_audit.append(
                {
                    "sample_id": row["id"],
                    "target": target,
                    "scenario": scenario,
                    "target_object": row["_visual_target_object"],
                    "declared_frames": expected_frames,
                    "decoded_frames": decoded_count,
                    "sampled_frame_indices": [index for index, _image in selected_frames],
                    "frame_count_matches_manifest": decoded_count == expected_frames,
                }
            )

        sheet_height = 8 + len(panels) * 330
        sheet = Image.new(
            "RGB", (GRID_COLUMNS * THUMBNAIL_SIZE[0], sheet_height), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for panel_index, (target, row, selected_frames, decoded_count) in enumerate(panels):
            y_base = 8 + panel_index * 330
            draw.text(
                (4, y_base),
                (
                    f"{target} | {row['id']} | "
                    f"declared={row['media'][0]['num_frames']} decoded={decoded_count}"
                ),
                fill="black",
                font=ImageFont.load_default(),
            )
            for position, (_frame_index, thumbnail) in enumerate(selected_frames):
                column, line = position % GRID_COLUMNS, position // GRID_COLUMNS
                sheet.paste(
                    thumbnail,
                    (column * THUMBNAIL_SIZE[0], y_base + 20 + line * THUMBNAIL_SIZE[1]),
                )
        sheet.save(output_dir / f"{scenario}.png")

    return {
        "frames_per_clip": VISUAL_FRAMES,
        "frame_audit": frame_audit,
        "contact_sheet_paths": [f"{scenario}.png" for scenario in sorted(by_scenario)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_train_rows(args.train_jsonl)
    summary = audit_train_rows(rows, args.corpus_root)
    selected = select_visual_review_rows(rows)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "visual-review-selection.jsonl"
    selection_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    frame_report = create_contact_sheets(
        selected, args.corpus_root, output / "contact-sheets"
    )
    report = {
        "status": "complete_train_only_physionpp_data_quality_audit",
        "train_manifest_sha256": sha256_file(args.train_jsonl),
        "summary": summary,
        "visual_selection_count": len(selected),
        "visual_selection_unit": "one Yes/No pair per scenario-target-object stratum",
        "visual_selection_sha256": sha256_file(selection_path),
        "visual_review": frame_report,
        "validation_or_sealed_data_opened": False,
        "model_loaded_or_trained": False,
    }
    (output / "audit-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
