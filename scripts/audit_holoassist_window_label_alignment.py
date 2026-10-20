"""Audit HoloAssist action-label timing against frozen 30-second windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ANNOTATION_SHA256 = "cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c"
SELECTION_SHA256 = "577adbafe6be869de3aff3b814acf91db7be582a42e0a7a83e7ee773dde0e8b7"
EVENT_HASHES = {
    "train": "bc4e34f87a8b4b3c8e493f3d1c60b90ed1ea8cf4f5438b0b8d263313c4f319b1",
    "validation": "1166c84b23720aa6239d085cad203c56224a0917bb3d8712b4448f5fd5d9555a",
}
PREDICTION_HASHES = {
    "audio_only": "8f2ee2111a95f9bd56e3cfabcc90a687a6aafc9c510d42deb822453a864a8138",
    "video_only": "54eef8a2d6c79c3bcf498d0a6b7ce1fac0960d945a487bd63ea4f4b3c14f6e1b",
    "joint": "5636107c8c630f55dbb409d36df04367b42892205aae986358776b6ab8e24323",
}
CHECKPOINT_HASHES = {
    "audio_only": "e14aa268e7773abda0e337a69edfc35f02e1547b5159a6f3b9759e4f0610119c",
    "video_only": "323d8f0b02e0fe27834c58abbef680cc5169b250eb45367b80e742ce7d1b51ce",
    "joint": "400e987c2312e14ddd764489e6eea3bf8541a29563a2c4a0dd2f04a959bc4047",
}
TRAINING_SOURCE_COMMIT = "327d63e61b279e293520e7afaef3f22294ff214b"
CONFIG_SHA256 = "f4de89195cbe5928a65a7b0ca634ee86ea3e25548856db07fbeca020d307ac32"
RUN_METADATA_SHA256 = "a96d4a873da134b02ed9269316360161f2a96a9b678750593607a60f913fff2f"
WINDOW_HASHES = {
    "train": "52e028767aa00c5d2effc2735a1a404d1a18a64a7ee1a39d947e1c59f7041b81",
    "validation": "b58c198d611ceab8d2d435004203ef8f5079a389e3560a41937118480c64c671",
}
WRONG_LABELS = {
    "Wrong Action, corrected by instructor verbally",
    "Wrong Action, corrected by student",
    "Wrong Action, not corrected",
}
VERBAL_CORRECTION = "Wrong Action, corrected by instructor verbally"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} hash differs from the pinned artifact")


def validate_run_metadata_document(metadata: dict) -> None:
    if (
        metadata.get("source_commit") != TRAINING_SOURCE_COMMIT
        or metadata.get("config_sha256") != CONFIG_SHA256
        or metadata.get("status") != "complete"
        or metadata.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("training run metadata differs from the fixed completed run")


def validate_run_metadata(path: Path) -> tuple[dict, str]:
    metadata_hash = sha256_file(path)
    _require_hash(metadata_hash, RUN_METADATA_SHA256, "training run metadata")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    validate_run_metadata_document(metadata)
    return metadata, metadata_hash


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_events(annotation: dict) -> dict[str, dict]:
    task_id = str(annotation.get("taskId", ""))
    result = {}
    for event in annotation.get("events", []):
        if event.get("label") == "Fine grained action":
            event_id = str(event.get("id", ""))
            if event_id:
                result[f"{task_id}:{event_id}"] = event
    return result


def _sequence_ids(value: object) -> set[str]:
    return {part.strip() for part in str(value).split(",") if part.strip()}


def audit_split(
    split: str,
    annotations_by_session: dict[str, dict],
    event_rows: list[dict],
    window_rows: list[dict],
) -> dict:
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation are permitted")
    if not event_rows or not window_rows:
        raise ValueError(f"{split} event/window rows cannot be empty")
    if any(row.get("split") != split for row in event_rows + window_rows):
        raise ValueError(f"split label mismatch in {split}")

    event_by_id = {str(row["id"]): row for row in event_rows}
    window_by_id = {str(row["id"]): row for row in window_rows}
    if len(event_by_id) != len(event_rows) or len(window_by_id) != len(window_rows):
        raise ValueError("duplicate event or window IDs")
    if {str(row["session_id"]) for row in event_rows} != set(annotations_by_session):
        raise ValueError("annotation session set does not match selected event sessions")

    source_by_event: dict[str, dict] = {}
    annotations_by_event: dict[str, dict] = {}
    subtype_counts: Counter[str] = Counter()
    wrong_event_ids: set[str] = set()
    for session_id, annotation in annotations_by_session.items():
        if annotation.get("video_name") != session_id:
            raise ValueError(f"annotation session identity mismatch: {session_id}")
        for event_id, source in _source_events(annotation).items():
            annotations_by_event[event_id] = annotation
            source_by_event[event_id] = source

    for event_id, row in event_by_id.items():
        source = source_by_event.get(event_id)
        if source is None:
            raise ValueError(f"event corpus ID is absent from source annotations: {event_id}")
        attrs = source.get("attributes", {})
        label = str(attrs.get("Action Correctness", ""))
        expected_target = 0 if label == "Correct Action" else 1 if label in WRONG_LABELS else None
        if expected_target is None or int(row["target"]) != expected_target:
            raise ValueError(f"source/corpus label differs for {event_id}")
        if (
            float(row["action_start_seconds"]) != float(source["start"])
            or float(row["action_end_seconds"]) != float(source["end"])
        ):
            raise ValueError(f"source/corpus interval differs for {event_id}")
        subtype_counts[label] += 1
        if expected_target:
            wrong_event_ids.add(event_id)

    events_by_session: dict[str, list[dict]] = defaultdict(list)
    windows_by_session: dict[str, list[dict]] = defaultdict(list)
    for row in event_rows:
        events_by_session[str(row["session_id"])].append(row)
    for row in window_rows:
        windows_by_session[str(row["session_id"])].append(row)
    if set(events_by_session) != set(windows_by_session):
        raise ValueError("event and window session sets differ")

    expected_members: dict[str, list[dict]] = {window_id: [] for window_id in window_by_id}
    for session_id, events in events_by_session.items():
        windows = sorted(windows_by_session[session_id], key=lambda row: int(row["window_index"]))
        for event in events:
            end = float(event["action_end_seconds"])
            matches = [
                window
                for window in windows
                if float(window["window_start_seconds"]) < end
                <= float(window["window_end_seconds"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"event does not map to exactly one temporal window: {event['id']}"
                )
            expected_members[str(matches[0]["id"])].append(event)

    cue_window_ids: set[str] = set()
    explicit_correction_window_ids: set[str] = set()
    linked_verbal_event_ids: set[str] = set()
    unlinked_verbal_event_ids: set[str] = set()
    for window_id, window in window_by_id.items():
        members = expected_members[window_id]
        members.sort(
            key=lambda row: (
                float(row["action_end_seconds"]),
                float(row["action_start_seconds"]),
                str(row["id"]),
            )
        )
        member_ids = [str(row["id"]) for row in members]
        if list(window.get("event_ids", [])) != member_ids:
            raise ValueError(f"window membership differs from source intervals: {window_id}")
        wrong = [event_id for event_id in member_ids if int(event_by_id[event_id]["target"]) == 1]
        expected_target = int(bool(wrong)) if members else None
        if window.get("target") != expected_target:
            raise ValueError(f"window target differs from member action labels: {window_id}")
        if bool(window.get("loss_mask")) != bool(members):
            raise ValueError(f"window loss mask differs from member actions: {window_id}")
        if (
            int(window.get("action_count", -1)) != len(members)
            or int(window.get("incorrect_action_count", -1)) != len(wrong)
            or int(window.get("correct_action_count", -1)) != len(members) - len(wrong)
        ):
            raise ValueError(f"window action counts differ from member labels: {window_id}")

        for event_id in wrong:
            source = source_by_event[event_id]
            attrs = source.get("attributes", {})
            if attrs.get("Action Correctness") != VERBAL_CORRECTION:
                continue
            correction_id = str(attrs.get("Incorrect Action Corrected by", ""))
            annotation = annotations_by_event[event_id]
            conversations = [
                event
                for event in annotation.get("events", [])
                if event.get("label") == "Conversation"
                and str(event.get("id")) == correction_id
            ]
            sequence_linked = [
                item
                for item in annotation.get("events", [])
                if item.get("label") == "Conversation"
                and str(source["id"])
                in _sequence_ids(item.get("attributes", {}).get("sequence", ""))
            ]
            correction_by_id = [
                item for item in conversations if str(item.get("id")) == correction_id
            ]
            associated = {str(item.get("id")): item for item in sequence_linked + correction_by_id}
            if not associated:
                unlinked_verbal_event_ids.add(event_id)
                continue
            linked_verbal_event_ids.add(event_id)
            start = float(window["window_start_seconds"])
            end = float(window["window_end_seconds"])
            for item in associated.values():
                overlaps_window = float(item["start"]) < end and float(item["end"]) > start
                if not overlaps_window:
                    continue
                cue_window_ids.add(window_id)
                attrs = item.get("attributes", {})
                sequence_matches = str(source["id"]) in _sequence_ids(attrs.get("sequence", ""))
                marked_as_correction = "correct the wrong action" in str(
                    attrs.get("Conversation Purpose", "")
                )
                if sequence_matches and marked_as_correction:
                    explicit_correction_window_ids.add(window_id)

    positive_windows = {
        window_id for window_id, row in window_by_id.items() if row.get("target") == 1
    }
    return {
        "session_count": len(annotations_by_session),
        "event_count": len(event_rows),
        "window_count": len(window_rows),
        "labeled_window_count": sum(bool(row.get("loss_mask")) for row in window_rows),
        "positive_window_count": len(positive_windows),
        "wrong_action_subtypes": dict(sorted(subtype_counts.items())),
        "wrong_actions_with_valid_verbal_correction_link": len(linked_verbal_event_ids),
        "wrong_actions_with_unresolved_verbal_correction_link": len(unlinked_verbal_event_ids),
        "positive_windows_with_linked_instructor_speech": len(positive_windows & cue_window_ids),
        "positive_windows_with_explicit_correction_speech": len(
            positive_windows & explicit_correction_window_ids
        ),
        "positive_windows_without_linked_instructor_speech": len(positive_windows - cue_window_ids),
        "positive_window_linked_speech_coverage": (
            len(positive_windows & cue_window_ids) / len(positive_windows)
            if positive_windows
            else None
        ),
        "linked_instructor_speech_window_ids": sorted(cue_window_ids),
        "explicit_correction_speech_window_ids": sorted(explicit_correction_window_ids),
    }


def _metric(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": None, "nll": None, "brier": None}
    correct = 0
    nll = 0.0
    brier = 0.0
    for row in rows:
        target = int(row["target"])
        prediction = int(row["prediction"])
        probabilities = [float(value) for value in row["probabilities"]]
        if len(probabilities) != 2 or any(not math.isfinite(value) for value in probabilities):
            raise ValueError(f"invalid binary probabilities for {row.get('id')}")
        if abs(sum(probabilities) - 1.0) > 1e-5 or min(probabilities) < 0 or max(probabilities) > 1:
            raise ValueError(f"probabilities are not normalized for {row.get('id')}")
        correct += int(prediction == target)
        nll -= math.log(max(probabilities[target], 1e-12))
        brier += sum((probabilities[index] - int(index == target)) ** 2 for index in range(2))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "nll": nll / len(rows),
        "brier": brier / len(rows),
    }


def summarize_prediction_strata(
    prediction_rows: list[dict],
    cue_window_ids: set[str],
    explicit_correction_window_ids: set[str],
) -> dict:
    by_id = {str(row["id"]): row for row in prediction_rows}
    if len(by_id) != len(prediction_rows):
        raise ValueError("duplicate prediction IDs")
    if not by_id:
        raise ValueError("prediction rows cannot be empty")
    for row in prediction_rows:
        if int(row["prediction"]) not in {0, 1} or int(row["target"]) not in {0, 1}:
            raise ValueError("prediction and target must be binary")
    groups = {
        "all": list(by_id.values()),
        "positive_with_linked_instructor_speech": [
            row for row in by_id.values() if int(row["target"]) == 1 and row["id"] in cue_window_ids
        ],
        "positive_without_linked_instructor_speech": [
            row
            for row in by_id.values()
            if int(row["target"]) == 1 and row["id"] not in cue_window_ids
        ],
        "positive_with_explicit_correction_speech": [
            row
            for row in by_id.values()
            if int(row["target"]) == 1 and row["id"] in explicit_correction_window_ids
        ],
        "positive_without_explicit_correction_speech": [
            row
            for row in by_id.values()
            if int(row["target"]) == 1 and row["id"] not in explicit_correction_window_ids
        ],
        "negative": [row for row in by_id.values() if int(row["target"]) == 0],
    }
    return {name: _metric(rows) for name, rows in groups.items()}


def audit_predictions(
    readout_root: Path,
    validation_rows: list[dict],
    cue_window_ids: set[str],
    explicit_correction_window_ids: set[str],
) -> dict:
    expected = {
        str(row["id"]): int(row["target"])
        for row in validation_rows
        if row.get("loss_mask")
    }
    result = {}
    for arm in ("audio_only", "video_only", "joint"):
        arm_dir = readout_root / arm
        best = json.loads((arm_dir / "best-checkpoint.json").read_text(encoding="utf-8"))
        step = int(best["step"])
        prediction_path = arm_dir / f"predictions-step-{step:06d}.json"
        prediction_hash = sha256_file(prediction_path)
        _require_hash(prediction_hash, PREDICTION_HASHES[arm], f"{arm} predictions")
        checkpoint_path = arm_dir / "best-checkpoint.pt"
        checkpoint_hash = sha256_file(checkpoint_path)
        _require_hash(checkpoint_hash, CHECKPOINT_HASHES[arm], f"{arm} best checkpoint")
        rows = json.loads(prediction_path.read_text(encoding="utf-8"))
        observed = {str(row["id"]): int(row["target"]) for row in rows}
        if observed != expected:
            raise ValueError(f"{arm} selected predictions do not match validation IDs/targets")
        result[arm] = {
            "selected_step": step,
            "prediction_sha256": prediction_hash,
            "best_checkpoint_sha256": checkpoint_hash,
            "strata": summarize_prediction_strata(
                rows, cue_window_ids, explicit_correction_window_ids
            ),
        }
    return result


def run_audit(
    *,
    annotations_path: Path,
    selection_path: Path,
    event_dir: Path,
    window_dir: Path,
    readout_root: Path,
    run_metadata_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_dir}")
    if sha256_file(annotations_path) != ANNOTATION_SHA256:
        raise ValueError("train/validation annotation file hash differs from frozen source")
    if sha256_file(selection_path) != SELECTION_SHA256:
        raise ValueError("selected-media manifest hash differs from frozen source")

    run_metadata, run_metadata_hash = validate_run_metadata(run_metadata_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selection"]
    selected_ids = {
        split: {str(row["session_id"]) for row in selection[split]["videos"]}
        for split in ("train", "validation")
    }
    if selected_ids["train"] & selected_ids["validation"]:
        raise ValueError("train/validation selected sessions overlap")

    annotations_raw = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations_by_split: dict[str, dict[str, dict]] = {"train": {}, "validation": {}}
    for annotation in annotations_raw:
        session_id = str(annotation.get("video_name", ""))
        for split in ("train", "validation"):
            if session_id in selected_ids[split]:
                if session_id in annotations_by_split[split]:
                    raise ValueError(f"duplicate selected annotation session: {session_id}")
                annotations_by_split[split][session_id] = annotation
    del annotations_raw

    outputs = {}
    file_hashes = {
        "annotations": sha256_file(annotations_path),
        "selection": sha256_file(selection_path),
        "run_metadata": run_metadata_hash,
    }
    cue_window_ids = set()
    explicit_correction_window_ids = set()
    loaded_windows = {}
    for split in ("train", "validation"):
        event_path = event_dir / f"{split}.jsonl"
        window_path = window_dir / f"{split}.jsonl"
        if sha256_file(event_path) != EVENT_HASHES[split]:
            raise ValueError(f"{split} event corpus hash differs from frozen receipt")
        if sha256_file(window_path) != WINDOW_HASHES[split]:
            raise ValueError(f"{split} temporal window hash differs from frozen receipt")
        events = _read_jsonl(event_path)
        windows = _read_jsonl(window_path)
        if {str(row["session_id"]) for row in events} != selected_ids[split]:
            raise ValueError(f"{split} event sessions differ from the immutable selection")
        if {str(row["session_id"]) for row in windows} != selected_ids[split]:
            raise ValueError(f"{split} window sessions differ from the immutable selection")
        result = audit_split(split, annotations_by_split[split], events, windows)
        outputs[split] = result
        loaded_windows[split] = windows
        if split == "validation":
            cue_window_ids.update(result["linked_instructor_speech_window_ids"])
            explicit_correction_window_ids.update(result["explicit_correction_speech_window_ids"])
        file_hashes[f"{split}_events"] = sha256_file(event_path)
        file_hashes[f"{split}_windows"] = sha256_file(window_path)

    validation_rows = loaded_windows["validation"]
    prediction_audit = audit_predictions(
        readout_root,
        validation_rows,
        cue_window_ids,
        explicit_correction_window_ids,
    )
    for arm in ("audio_only", "video_only", "joint"):
        metrics = prediction_audit[arm]["strata"]
        if metrics["all"]["n"] != sum(bool(row.get("loss_mask")) for row in validation_rows):
            raise ValueError(f"{arm} selected prediction count differs from validation windows")
        best_path = readout_root / arm / "best-checkpoint.pt"
        file_hashes[f"{arm}_best_checkpoint"] = sha256_file(best_path)
        file_hashes[f"{arm}_selected_predictions"] = prediction_audit[arm]["prediction_sha256"]

    receipt = {
        "status": "PASS_HOLOASSIST_WINDOW_LABEL_ALIGNMENT_AUDIT",
        "scope": "post-hoc source-label and temporal-window audit; train/validation only",
        "selection_sha256": SELECTION_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "training_source_commit": run_metadata["source_commit"],
        "training_config_sha256": run_metadata["config_sha256"],
        "training_started_at_utc": run_metadata["started_at_utc"],
        "training_finished_at_utc": run_metadata["finished_at_utc"],
        "training_device": run_metadata["device"],
        "source_files_sha256": file_hashes,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "splits": outputs,
        "selected_checkpoint_prediction_strata": prediction_audit,
        "test_or_sealed_data_opened": False,
        "media_opened": False,
        "model_loaded": False,
        "training": False,
        "gpu_used": False,
        "interpretation_warning": (
            "Verbal-correction speech is a valid in-window observation but is a direct "
            "label-correlated cue for the annotated error subtype. These are selected-validation "
            "post-hoc strata, not independent or causal evidence."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["result_sha256"] = sha256_file(result_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--readout-root", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_audit(
        annotations_path=args.annotations,
        selection_path=args.selection,
        event_dir=args.event_dir,
        window_dir=args.window_dir,
        readout_root=args.readout_root,
        run_metadata_path=args.run_metadata,
        output_dir=args.output,
    )
    print(json.dumps({
        "status": receipt["status"],
        "splits": {
            split: {
                key: value
                for key, value in report.items()
                if key not in {
                    "linked_instructor_speech_window_ids",
                    "explicit_correction_speech_window_ids",
                }
            }
            for split, report in receipt["splits"].items()
        },
        "selected_checkpoint_prediction_strata": receipt[
            "selected_checkpoint_prediction_strata"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

