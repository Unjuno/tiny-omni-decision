"""Summarize saved Perception Test AV errors by hidden source-label stratum."""

from __future__ import annotations

import argparse
import json
import platform
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tiny_omni_decision.dataset import sha256_file

EXPECTED_COHORT_SHA256 = "dd8c9de6d8970eb1105cb81551e386fca563eb174b8bcda5b09dcc8a16e567be"
EXPECTED_ANNOTATIONS_SHA256 = "4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141"
EXPECTED_TRAIN_ANNOTATIONS_SHA256 = (
    "0200a50644ab9d36a2dd80e5239e709cabeeaf2e4274eca201c90743b89cc6f5"
)
EXPECTED_READOUT_STATUS = "PASS_MATCHED_FROZEN_AV_READOUT"
MODELS = ("audio", "video", "joint")


def summarize_strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "examples": 0,
            "visible": 0,
            "not_visible": 0,
            "correct": {model: 0 for model in MODELS},
        }
    )
    for row in rows:
        label = row["source_label"]
        target = row["target"]
        if target not in ("visible", "not_visible"):
            raise ValueError("unknown visibility target")
        if any(row[f"{model}_prediction"] not in ("visible", "not_visible") for model in MODELS):
            raise ValueError("unknown model prediction")
        summary = strata[label]
        summary["examples"] += 1
        summary[target] += 1
        for model in MODELS:
            summary["correct"][model] += int(row[f"{model}_prediction"] == target)

    by_label = {}
    for label, summary in sorted(strata.items(), key=lambda pair: (-pair[1]["examples"], pair[0])):
        count = summary["examples"]
        by_label[label] = {
            **summary,
            "contains_both_targets": summary["visible"] > 0 and summary["not_visible"] > 0,
            "accuracy": {model: summary["correct"][model] / count for model in MODELS},
        }
    return {"examples": len(rows), "source_label_strata": by_label}


def summarize_source_support(annotation_rows: dict[str, Any]) -> dict[str, Any]:
    """Count label support by event label, keeping videos as the group unit."""
    strata: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "segments": 0,
            "videos": 0,
            "visible_segments": 0,
            "not_visible_segments": 0,
            "visible_videos": 0,
            "not_visible_videos": 0,
            "videos_with_both_labels": 0,
        }
    )
    for _video_id, video in annotation_rows.items():
        labels_by_video: dict[str, set[bool]] = defaultdict(set)
        for segment in video["sound_localisation"]:
            label = segment["label"]
            visible = bool(segment["is_visible"])
            summary = strata[label]
            summary["segments"] += 1
            summary["visible_segments" if visible else "not_visible_segments"] += 1
            labels_by_video[label].add(visible)
        for label, labels in labels_by_video.items():
            summary = strata[label]
            summary["videos"] += 1
            summary["visible_videos"] += int(True in labels)
            summary["not_visible_videos"] += int(False in labels)
            summary["videos_with_both_labels"] += int(len(labels) == 2)
    return dict(
        videos=len(annotation_rows),
        source_label_strata={
            label: summary
            for label, summary in sorted(
                strata.items(), key=lambda pair: (-pair[1]["segments"], pair[0])
            )
        },
    )


def run(
    cohort_path: Path,
    annotations_path: Path,
    train_annotations_path: Path,
    readout_dir: Path,
) -> dict[str, Any]:
    cohort_path = cohort_path.resolve(strict=True)
    annotations_path = annotations_path.resolve(strict=True)
    train_annotations_path = train_annotations_path.resolve(strict=True)
    readout_dir = readout_dir.resolve(strict=True)
    cohort_sha = sha256_file(cohort_path)
    annotations_sha = sha256_file(annotations_path)
    train_annotations_sha = sha256_file(train_annotations_path)
    if (
        cohort_sha != EXPECTED_COHORT_SHA256
        or annotations_sha != EXPECTED_ANNOTATIONS_SHA256
        or train_annotations_sha != EXPECTED_TRAIN_ANNOTATIONS_SHA256
    ):
        raise ValueError("cohort or validation annotation hash differs from pinned input")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    result_path = readout_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        cohort.get("train_validation_video_overlap") != 0
        or result.get("status") != EXPECTED_READOUT_STATUS
        or result.get("cohort_sha256") != cohort_sha
    ):
        raise ValueError("readout or split integrity receipt mismatch")

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        path = readout_dir / f"{model}_validation_predictions.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        if len(records) != 256 or len({row["video_id"] for row in records}) != len(records):
            raise ValueError(f"invalid {model} prediction cohort")
        predictions[model] = {row["video_id"]: row for row in records}

    with zipfile.ZipFile(annotations_path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError("unexpected validation annotation archive layout")
        annotation_rows = json.loads(archive.read(names[0]))
    with zipfile.ZipFile(train_annotations_path) as archive:
        names = archive.namelist()
        if names != ["sound_localisation_train.json"]:
            raise ValueError("unexpected training annotation archive layout")
        train_annotation_rows = json.loads(archive.read(names[0]))

    joined = []
    cohort_rows = cohort["splits"]["valid"]
    if len(cohort_rows) != 256 or len({row["video_id"] for row in cohort_rows}) != 256:
        raise ValueError("unexpected pinned validation cohort size or repeated video")
    for event in cohort_rows:
        video_id, event_id = event["video_id"], event["event_id"]
        candidates = [
            row
            for row in annotation_rows[video_id]["sound_localisation"]
            if row["id"] == event_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"annotation event is missing or duplicated: {video_id}:{event_id}")
        annotation = candidates[0]
        target = "visible" if event["target"] else "not_visible"
        if bool(annotation["is_visible"]) != bool(event["target"]):
            raise ValueError(f"cohort target mismatch: {video_id}:{event_id}")
        predictions_for_event = {}
        for model in MODELS:
            prediction = predictions[model].get(video_id)
            if prediction is None or prediction["event_id"] != event_id:
                raise ValueError(f"prediction/event identity mismatch: {model}:{video_id}")
            if prediction["target"] != target:
                raise ValueError(f"prediction target mismatch: {model}:{video_id}")
            predictions_for_event[f"{model}_prediction"] = prediction["prediction"]
        joined.append(
            {
                "video_id": video_id,
                "event_id": event_id,
                "source_label": annotation["label"],
                "target": target,
                **predictions_for_event,
            }
        )

    summary = summarize_strata(joined)
    return {
        "schema_version": 1,
        "status": "POSTHOC_VALIDATION_ERROR_STRATA_ONLY",
        "cohort_sha256": cohort_sha,
        "validation_annotations_sha256": annotations_sha,
        "train_annotations_sha256": train_annotations_sha,
        "readout_result_sha256": sha256_file(result_path),
        "prediction_sha256": {
            model: sha256_file(readout_dir / f"{model}_validation_predictions.json")
            for model in MODELS
        },
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "readout_status": result["status"],
        "models": list(MODELS),
        "metrics": summary,
        "source_label_support": {
            "train": summarize_source_support(train_annotation_rows),
            "official_validation": summarize_source_support(annotation_rows),
        },
        "interpretation_limit": (
            "The source label is posthoc annotation metadata and was not a model input. "
            "Strata are small, come from a previously inspected balanced validation cohort, "
            "and must not be used for model selection, sample filtering, or product claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    report = run(args.cohort, args.annotations, args.train_annotations, args.readout_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
