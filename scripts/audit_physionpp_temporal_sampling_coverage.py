"""Measure fixed frame-sampling coverage of official Physion++ train events."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_SHA256 = "0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b"
DEFAULT_OFFICIAL_RESULTS_SHA256 = (
    "354175780c48929c1be1c137bd76b4af0520f10605a3ed1b75237c350068dfdb"
)
FRAME_POLICIES = (8, 16, 32, 64)
TOLERANCES = (0, 1, 2, 4)


def sample_indices(frame_count: int, sample_count: int) -> tuple[int, ...]:
    if frame_count < 2 or sample_count < 2 or sample_count > frame_count:
        raise ValueError("frame_count and sample_count are incompatible")
    indices = tuple(
        round(i * (frame_count - 1) / (sample_count - 1))
        for i in range(sample_count)
    )
    if len(set(indices)) != sample_count:
        raise ValueError("sampling policy produced duplicate frame indices")
    return indices


def contiguous_runs(frame_indices: list[int]) -> int:
    if frame_indices != sorted(set(frame_indices)):
        raise ValueError("frame indices must be sorted and unique")
    if not frame_indices:
        return 0
    return 1 + sum(
        current != previous + 1
        for previous, current in zip(frame_indices[:-1], frame_indices[1:], strict=True)
    )


def coverage_counts(
    frame_count: int,
    positive_frames: list[int],
    sample_count: int,
    tolerance: int,
) -> tuple[bool, int]:
    samples = sample_indices(frame_count, sample_count)
    covered = [
        event_frame
        for event_frame in positive_frames
        if any(abs(event_frame - sample) <= tolerance for sample in samples)
    ]
    return bool(covered), len(covered)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sampling_audit(
    train_jsonl: Path,
    official_results_jsonl: Path,
    output_dir: Path,
    *,
    expected_train_sha256: str = DEFAULT_TRAIN_SHA256,
    expected_official_results_sha256: str = DEFAULT_OFFICIAL_RESULTS_SHA256,
) -> dict[str, Any]:
    train_sha = _sha256(train_jsonl)
    official_sha = _sha256(official_results_jsonl)
    if train_sha != expected_train_sha256.lower():
        raise ValueError(f"train manifest SHA-256 mismatch: {train_sha}")
    if official_sha != expected_official_results_sha256.lower():
        raise ValueError(f"official result SHA-256 mismatch: {official_sha}")

    train_rows = [
        json.loads(line)
        for line in train_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    official_rows = [
        json.loads(line)
        for line in official_results_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_by_id = {row.get("id"): row for row in train_rows}
    official_by_id = {row.get("sample_id"): row for row in official_rows}
    if (
        not train_rows
        or any(row.get("split") != "train" for row in train_rows)
        or len(train_by_id) != len(train_rows)
        or set(train_by_id) != set(official_by_id)
    ):
        raise ValueError("train and official results must have identical unique train IDs")

    per_row: list[dict[str, Any]] = []
    scenario_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for train_row in train_rows:
        official = official_by_id[train_row["id"]]
        media = train_row.get("media", [{}])[0]
        frame_count = media.get("num_frames")
        contact_frames = official.get("positive_contact_frames")
        if (
            not isinstance(frame_count, int)
            or frame_count != official.get("frame_count")
            or not isinstance(contact_frames, list)
            or any(
                type(index) is not int or index < 0 or index >= frame_count
                for index in contact_frames
            )
            or contact_frames != sorted(set(contact_frames))
            or official.get("frame_timeline_match") is not True
            or official.get("label_match") is not True
            or official.get("stimulus_identity_match") is not True
        ):
            raise ValueError(f"invalid official timeline for {train_row['id']}")
        positive = train_row.get("target") == "Yes"
        if positive != bool(contact_frames):
            raise ValueError(f"official timeline/label disagreement for {train_row['id']}")
        if official.get("first_contact_frame") != (
            contact_frames[0] if contact_frames else None
        ):
            raise ValueError(f"first-contact frame mismatch for {train_row['id']}")

        scenario = train_row["source_record_id"].split("/")[1]
        row_result: dict[str, Any] = {
            "sample_id": train_row["id"],
            "scenario": scenario,
            "target": train_row["target"],
            "frame_count": frame_count,
            "positive_contact_frame_count": len(contact_frames),
            "contact_run_count": contiguous_runs(contact_frames),
            "contact_frame_fraction": len(contact_frames) / frame_count,
            "first_contact_fraction": (
                contact_frames[0] / max(1, frame_count - 1) if contact_frames else None
            ),
        }
        for sample_count in FRAME_POLICIES:
            for tolerance in TOLERANCES:
                video_hit, covered_event_frames = coverage_counts(
                    frame_count, contact_frames, sample_count, tolerance
                )
                row_result[f"frames_{sample_count}_tol_{tolerance}_video_hit"] = video_hit
                row_result[f"frames_{sample_count}_tol_{tolerance}_covered_events"] = (
                    covered_event_frames
                )
        per_row.append(row_result)
        scenario_groups[scenario].append(row_result)

    positives = [row for row in per_row if row["target"] == "Yes"]
    if not positives:
        raise ValueError("train split contains no positive contact events")

    policy_summary: list[dict[str, Any]] = []
    for sample_count in FRAME_POLICIES:
        for tolerance in TOLERANCES:
            covered_videos = sum(
                row[f"frames_{sample_count}_tol_{tolerance}_video_hit"]
                for row in positives
            )
            covered_frames = sum(
                row[f"frames_{sample_count}_tol_{tolerance}_covered_events"]
                for row in positives
            )
            total_event_frames = sum(row["positive_contact_frame_count"] for row in positives)
            policy_summary.append(
                {
                    "sampled_frames": sample_count,
                    "tolerance_frames": tolerance,
                    "positive_videos": len(positives),
                    "videos_with_any_sample_near_event": covered_videos,
                    "video_capture_rate": covered_videos / len(positives),
                    "covered_labeled_event_frames": covered_frames,
                    "total_labeled_event_frames": total_event_frames,
                    "event_frame_recall": covered_frames / total_event_frames,
                }
            )

    scenario_summary: list[dict[str, Any]] = []
    for scenario, scenario_rows in sorted(scenario_groups.items()):
        scenario_positives = [row for row in scenario_rows if row["target"] == "Yes"]
        summary: dict[str, Any] = {
            "scenario": scenario,
            "train_videos": len(scenario_rows),
            "positive_videos": len(scenario_positives),
            "total_positive_contact_frames": sum(
                row["positive_contact_frame_count"] for row in scenario_positives
            ),
        }
        for sample_count in FRAME_POLICIES:
            for tolerance in (0, 2, 4):
                hits = sum(
                    row[f"frames_{sample_count}_tol_{tolerance}_video_hit"]
                    for row in scenario_positives
                )
                summary[f"frames_{sample_count}_tol_{tolerance}_video_capture_rate"] = (
                    hits / len(scenario_positives) if scenario_positives else None
                )
        scenario_summary.append(summary)

    output_dir.mkdir(parents=True)
    row_path = output_dir / "per-video-sampling-coverage.jsonl"
    row_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_row),
        encoding="utf-8",
    )
    durations = [row["contact_frame_fraction"] for row in positives]
    first_contact_fractions = [row["first_contact_fraction"] for row in positives]
    report = {
        "status": "pass",
        "purpose": (
            "train-only exact official contact-timeline sampling audit; "
            "not model quality or proof of visual observability"
        ),
        "train_manifest_sha256": train_sha,
        "official_results_sha256": official_sha,
        "train_videos": len(train_rows),
        "positive_videos": len(positives),
        "negative_videos": len(train_rows) - len(positives),
        "frame_policies": list(FRAME_POLICIES),
        "tolerances_frames": list(TOLERANCES),
        "sampling_policy": "round(i*(frame_count-1)/(sample_count-1)); endpoints included",
        "policy_summary": policy_summary,
        "scenario_summary": scenario_summary,
        "positive_event_statistics": {
            "total_positive_contact_frames": sum(
                row["positive_contact_frame_count"] for row in positives
            ),
            "videos_with_multiple_contiguous_contact_runs": sum(
                row["contact_run_count"] > 1 for row in positives
            ),
            "median_contact_frame_fraction": statistics.median(durations),
            "median_first_contact_fraction": statistics.median(first_contact_fractions),
        },
        "per_video_results_sha256": _sha256(row_path),
        "validation_or_sealed_data_opened": False,
        "media_or_model_loaded": False,
        "gpu_used": False,
    }
    (output_dir / "audit-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--official-results-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", default=DEFAULT_TRAIN_SHA256)
    parser.add_argument(
        "--expected-official-results-sha256", default=DEFAULT_OFFICIAL_RESULTS_SHA256
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_sampling_audit(
                args.train_jsonl,
                args.official_results_jsonl,
                args.output_dir,
                expected_train_sha256=args.expected_train_sha256,
                expected_official_results_sha256=args.expected_official_results_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
