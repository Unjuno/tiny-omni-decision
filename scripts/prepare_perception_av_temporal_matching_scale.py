"""Prepare a fresh, group-disjoint AV temporal-matching scale comparison."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.prepare_perception_av_temporal_matching import (
    SEED,
    SOURCE_REVISION,
    _make_trial,
    eligible_pairs,
    load_annotation_archive,
    sha256_file,
    stable_rank,
)


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stable_row_key(seed: int, label_id: int, row: dict[str, Any]) -> str:
    return stable_rank(
        seed, "perception-av-temporal-scale-v2", label_id, row["video_id"], _canonical(row)
    )


def _allocate_counts(
    by_label: dict[int, list[dict[str, Any]]], total: int, seed: int
) -> dict[int, int]:
    population = sum(len(rows) for rows in by_label.values())
    if total < 0 or total > population:
        raise ValueError(f"requested {total} rows from a population of {population}")
    if population == 0:
        if total == 0:
            return {label: 0 for label in by_label}
        raise ValueError("cannot allocate from empty label buckets")

    raw = {label: total * len(rows) / population for label, rows in by_label.items()}
    counts = {label: int(value) for label, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        by_label,
        key=lambda label: (-(raw[label] - counts[label]), stable_rank(seed, "label-tie", label)),
    )
    for label in order[:remainder]:
        counts[label] += 1
    return counts


def select_grouped_cohorts(
    candidates_by_label: dict[int, list[dict[str, Any]]],
    *,
    train_count: int,
    small_train_count: int,
    validation_count: int,
    seed: int = SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Select nested train cohorts and a separate validation set by video group."""
    total = train_count + validation_count
    if small_train_count < 1 or small_train_count > train_count:
        raise ValueError("small_train_count must be between 1 and train_count")
    queues: dict[int, list[dict[str, Any]]] = {}
    for label_id, candidates in candidates_by_label.items():
        unique: dict[str, dict[str, Any]] = {}
        for row in candidates:
            video_id = str(row["video_id"])
            if int(row["label_id"]) != int(label_id):
                raise ValueError("candidate label does not match its bucket")
            current = unique.get(video_id)
            if current is None or _stable_row_key(seed, label_id, row) < _stable_row_key(
                seed, label_id, current
            ):
                unique[video_id] = row
        queues[int(label_id)] = sorted(
            unique.values(), key=lambda row: _stable_row_key(seed, int(label_id), row)
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    cursors = dict.fromkeys(queues, 0)
    while len(selected) < total:
        progressed = False
        for label_id in sorted(queues):
            queue = queues[label_id]
            while cursors[label_id] < len(queue):
                candidate = queue[cursors[label_id]]
                cursors[label_id] += 1
                video_id = str(candidate["video_id"])
                if video_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(video_id)
                progressed = True
                break
            if len(selected) == total:
                break
        if not progressed:
            raise ValueError(
                f"requested {total} rows but only {len(selected)} unique video groups are available"
            )

    selected_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_label[int(row["label_id"])].append(row)
    for label_id, rows in selected_by_label.items():
        rows.sort(key=lambda row: _stable_row_key(seed, label_id, row))

    train_allocation = _allocate_counts(selected_by_label, train_count, seed)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_by_label: dict[int, list[dict[str, Any]]] = {}
    for label_id, rows in selected_by_label.items():
        count = train_allocation[label_id]
        train_by_label[label_id] = rows[:count]
        train_rows.extend(rows[:count])
        validation_rows.extend(rows[count:])

    small_allocation = _allocate_counts(train_by_label, small_train_count, seed)
    small_rows = [
        row
        for label_id, rows in sorted(train_by_label.items())
        for row in rows[: small_allocation[label_id]]
    ]
    train_rows.sort(key=lambda row: str(row["video_id"]))
    small_rows.sort(key=lambda row: str(row["video_id"]))
    validation_rows.sort(key=lambda row: str(row["video_id"]))

    train_ids = {str(row["video_id"]) for row in train_rows}
    small_ids = {str(row["video_id"]) for row in small_rows}
    validation_ids = {str(row["video_id"]) for row in validation_rows}
    if len(train_ids) != train_count or len(validation_ids) != validation_count:
        raise RuntimeError("selected cohort counts do not match the requested counts")
    if not small_ids <= train_ids:
        raise RuntimeError("small training cohort is not nested in the large cohort")
    if train_ids & validation_ids:
        raise RuntimeError("video group crosses train and validation")
    return {"train": train_rows, "train_small": small_rows, "validation": validation_rows}


_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


def _collect_candidate_ids(value: Any, candidate_ids: set[str], found: set[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                for token in _TOKEN_RE.findall(key):
                    if token in candidate_ids:
                        found.add(token)
            if key in {"video_id", "source_video_id", "group_id"} and isinstance(
                nested, (str, int)
            ):
                candidate_id = str(nested)
                if candidate_id in candidate_ids:
                    found.add(candidate_id)
            _collect_candidate_ids(nested, candidate_ids, found)
    elif isinstance(value, list):
        for nested in value:
            _collect_candidate_ids(nested, candidate_ids, found)
    elif isinstance(value, str):
        for token in _TOKEN_RE.findall(value):
            if token in candidate_ids:
                found.add(token)


def collect_previously_exposed_ids(
    ledger_path: Path, artifact_root: Path, candidate_ids: set[str]
) -> tuple[set[str], int]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("status") != "PASS_HASH_VERIFIED_PRIOR_EXPOSURE_IDENTITY_EXCLUSION":
        raise ValueError("prior exposure ledger did not pass its identity audit")
    found: set[str] = set()
    receipts = ledger.get("file_receipts")
    if not isinstance(receipts, list) or len(receipts) != ledger.get("file_count"):
        raise ValueError("prior exposure receipt count mismatch")
    root = artifact_root.resolve(strict=True)
    for receipt in receipts:
        relative = Path(receipt["path"])
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("exposure receipt escapes artifact root")
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"prior exposure hash mismatch: {relative}")
        if path.suffix not in {".json", ".jsonl"}:
            raise ValueError(f"unsupported exposure receipt format: {relative}")
        if path.suffix == ".json":
            _collect_candidate_ids(
                json.loads(path.read_text(encoding="utf-8")), candidate_ids, found
            )
        else:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        _collect_candidate_ids(json.loads(line), candidate_ids, found)
    return found, len(receipts)


def _manifest_video_ids(path: Path) -> set[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = set()
    for split in ("train", "train_small", "validation", "valid"):
        for row in manifest.get(split, []):
            video_id = row.get("video_id", row.get("group_id"))
            if video_id is not None:
                result.add(str(video_id))
    return result


def build_scale_manifest(
    *,
    annotations_path: Path,
    exposure_ledger_path: Path,
    artifact_root: Path,
    previously_used_manifests: list[Path],
    train_count: int = 224,
    small_train_count: int = 96,
    validation_count: int = 64,
) -> dict[str, Any]:
    annotations = load_annotation_archive(annotations_path)
    source_video_ids = {str(video_id) for video_id in annotations}
    exposed_ids, receipt_count = collect_previously_exposed_ids(
        exposure_ledger_path, artifact_root, source_video_ids
    )
    used_ids = set(exposed_ids)
    for path in previously_used_manifests:
        used_ids.update(_manifest_video_ids(path))

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for video_id, record in annotations.items():
        video_id = str(video_id)
        if video_id in used_ids:
            continue
        best_by_label: dict[int, tuple[str, dict[str, Any]]] = {}
        for pair in eligible_pairs(video_id, record):
            label_id = int(pair["label_id"])
            rank = stable_rank(
                SEED,
                "perception-av-temporal-scale-v2-pair",
                video_id,
                pair["anchor"]["event_id"],
                pair["partner"]["event_id"],
            )
            if label_id not in best_by_label or rank < best_by_label[label_id][0]:
                best_by_label[label_id] = (rank, pair)
        for label_id, (_, pair) in best_by_label.items():
            by_label[label_id].append({"video_id": video_id, "label_id": label_id, "pair": pair})

    cohorts = select_grouped_cohorts(
        by_label,
        train_count=train_count,
        small_train_count=small_train_count,
        validation_count=validation_count,
        seed=SEED,
    )
    train_rows = [_make_trial(row["pair"], split="train") for row in cohorts["train"]]
    small_rows = [_make_trial(row["pair"], split="train") for row in cohorts["train_small"]]
    validation_rows = [_make_trial(row["pair"], split="valid") for row in cohorts["validation"]]

    def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(str(row["label_id"]) for row in rows).items()))

    input_hashes = {
        "train_annotations": sha256_file(annotations_path),
        "prior_exposure_ledger": sha256_file(exposure_ledger_path),
        "previously_used_manifests": {
            str(path): sha256_file(path) for path in previously_used_manifests
        },
    }
    return {
        "schema_version": 1,
        "status": "PASS_FRESH_AV_TEMPORAL_SCALE_MANIFEST_PREFLIGHT",
        "task_id": "perception-av-same-video-same-label-temporal-match-scale-v2",
        "source_revision": SOURCE_REVISION,
        "seed": SEED,
        "source_split": "official train only",
        "task_claim": (
            "Development task: choose the synchronized audio window from two "
            "same-video, same-label windows for an eight-frame video interval."
        ),
        "split_policy": (
            "source video groups are disjoint; 96-train subset is nested inside "
            "224-train cohort; validation is 64 additional source videos"
        ),
        "selection_inputs_sha256": input_hashes,
        "verified_prior_exposure_receipts": receipt_count,
        "eligible_video_label_counts_before_cohort_selection": {
            str(label): len(rows) for label, rows in sorted(by_label.items())
        },
        "train": train_rows,
        "train_small": small_rows,
        "validation": validation_rows,
        "summary": {
            "unique_video_groups_available": len(
                set().union(
                    *(set(str(row["video_id"]) for row in rows) for rows in by_label.values())
                )
            ),
            "train_groups": len(train_rows),
            "small_train_groups": len(small_rows),
            "validation_groups": len(validation_rows),
            "train_by_label": label_counts(train_rows),
            "small_train_by_label": label_counts(small_rows),
            "validation_by_label": label_counts(validation_rows),
            "small_train_is_subset_of_train": True,
            "train_validation_video_overlap": 0,
            "prior_exposure_video_overlap": 0,
            "official_test_opened": False,
            "sealed_audit_opened": False,
            "validation_scope": (
                "fresh video-group development holdout from official train; "
                "checkpoint selection may use it; not blind or independent test"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--used-manifest", type=Path, action="append", default=[])
    parser.add_argument("--train-count", type=int, default=224)
    parser.add_argument("--small-train-count", type=int, default=96)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    result = build_scale_manifest(
        annotations_path=args.train_annotations.resolve(strict=True),
        exposure_ledger_path=args.exposure_ledger.resolve(strict=True),
        artifact_root=args.artifact_root.resolve(strict=True),
        previously_used_manifests=[path.resolve(strict=True) for path in args.used_manifest],
        train_count=args.train_count,
        small_train_count=args.small_train_count,
        validation_count=args.validation_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
