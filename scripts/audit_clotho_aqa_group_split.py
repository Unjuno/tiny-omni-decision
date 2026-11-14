"""Create a metadata-only, uploader-grouped Clotho-AQA development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALLOWED_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/3.0/",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def parse_binary_vote(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "yes":
            return 1
        if normalized == "no":
            return 0
        raise ValueError("votes must be exact yes/no strings or binary integers")
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError("votes must be exact yes/no strings or binary integers")


def build_split(
    rows: list[dict[str, Any]], *, seed: int, dev_groups: int, calibration_groups: int
) -> tuple[dict[str, Any], list[dict[str, str | int]]]:
    """Assign whole uploader groups by a stable hash; never stratify on labels."""
    if not rows:
        raise ValueError("input must contain at least one row")

    uploaders = sorted({str(row["uploader"]) for row in rows})
    if dev_groups <= 0 or calibration_groups <= 0:
        raise ValueError("dev_groups and calibration_groups must be positive")
    if dev_groups + calibration_groups >= len(uploaders):
        raise ValueError("the requested held-out groups leave no training groups")

    group_order = sorted(
        uploaders,
        key=lambda uploader: sha256_text(f"seed={seed}|clotho-aqa|uploader|{uploader}"),
    )
    assignment = {
        uploader: "dev" if index < dev_groups else "calibration"
        if index < dev_groups + calibration_groups
        else "train"
        for index, uploader in enumerate(group_order)
    }

    seen_samples: set[str] = set()
    file_groups: dict[str, str] = {}
    source_groups: dict[str, str] = {}
    split_rows: list[dict[str, str | int]] = []
    questions_by_split: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    group_counts: Counter[str] = Counter()
    file_counts: dict[str, set[str]] = defaultdict(set)
    disagreement_counts: Counter[str] = Counter()

    for row in rows:
        required = ("file_name", "source_id", "uploader", "question", "target", "votes")
        if any(key not in row for key in required):
            raise ValueError("row is missing a required source field")
        uploader = str(row["uploader"])
        file_name = str(row["file_name"])
        source_id = str(row["source_id"])
        if file_groups.setdefault(file_name, uploader) != uploader:
            raise ValueError("one audio filename maps to multiple uploader groups")
        if source_groups.setdefault(source_id, uploader) != uploader:
            raise ValueError("one source ID maps to multiple uploader groups")
        target = int(row["target"])
        votes = [parse_binary_vote(vote) for vote in row["votes"]]
        if target not in (0, 1) or len(votes) != 3 or any(vote not in (0, 1) for vote in votes):
            raise ValueError("rows must retain three binary votes and a binary majority target")
        if target != int(sum(votes) >= 2):
            raise ValueError("majority target does not match the three retained votes")
        if str(row.get("license", "")) not in ALLOWED_LICENSES:
            raise ValueError("row license is not one of the admitted exact clip licenses")

        split = assignment[uploader]
        question_id = sha256_text(normalize_question(str(row["question"])))
        file_id = sha256_text(file_name)
        sample_id = sha256_text(f"{source_id}|{question_id}")
        if sample_id in seen_samples:
            raise ValueError("duplicate audio-question sample identity")
        seen_samples.add(sample_id)

        split_rows.append(
            {
                "sample_id": sample_id,
                "audio_asset_id": file_id,
                "source_id": sha256_text(source_id),
                "group_id": sha256_text(f"clotho-aqa|uploader|{uploader}"),
                "normalized_question_id": question_id,
                "split": split,
                "target": target,
                "yes_votes": sum(votes),
                "vote_count": len(votes),
            }
        )
        questions_by_split[split].add(question_id)
        row_counts[split] += 1
        label_counts[split][target] += 1
        group_counts[split] += 1
        file_counts[split].add(file_id)
        disagreement_counts[split] += len(set(votes)) > 1

    group_ids_by_split: dict[str, set[str]] = defaultdict(set)
    files_by_split: dict[str, set[str]] = defaultdict(set)
    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for entry in split_rows:
        split = str(entry["split"])
        group_ids_by_split[split].add(str(entry["group_id"]))
        files_by_split[split].add(str(entry["audio_asset_id"]))
        sources_by_split[split].add(str(entry["source_id"]))

    split_names = ("train", "calibration", "dev")
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if group_ids_by_split[left] & group_ids_by_split[right]:
                raise ValueError("uploader group overlap across splits")
            if files_by_split[left] & files_by_split[right]:
                raise ValueError("audio filename overlap across splits")
            if sources_by_split[left] & sources_by_split[right]:
                raise ValueError("source ID overlap across splits")

    repeated_question_hashes = set.intersection(
        *(questions_by_split[name] for name in split_names)
    ) if all(questions_by_split[name] for name in split_names) else set()
    # Pairwise overlap is reported; a question appearing in only two splits also matters.
    question_overlap = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            question_overlap[f"{left}:{right}"] = len(
                questions_by_split[left] & questions_by_split[right]
            )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "assignment": "SHA-256(seed, dataset, uploader); sorted hash rank; no label stratification",
        "group_field": "uploader",
        "requested_groups": {
            "train": len(uploaders) - dev_groups - calibration_groups,
            "calibration": calibration_groups,
            "dev": dev_groups,
        },
        "splits": {
            name: {
                "groups": len(group_ids_by_split[name]),
                "audio_files": len(files_by_split[name]),
                "rows": row_counts[name],
                "target_counts": {str(k): label_counts[name][k] for k in (0, 1)},
                "questions_with_vote_disagreement": disagreement_counts[name],
            }
            for name in split_names
        },
        "cross_split_uploader_overlap": 0,
        "cross_split_audio_filename_overlap": 0,
        "cross_split_source_id_overlap": 0,
        "cross_split_normalized_question_overlap": question_overlap,
        "questions_repeated_in_all_three_splits": len(repeated_question_hashes),
        "media_downloaded": False,
        "waveform_hash_overlap_checked": False,
        "validation_test_or_sealed_audit_opened": False,
        "model_loaded_or_run": False,
        "training_started": False,
        "quality_metrics_computed": False,
    }
    split_rows.sort(key=lambda entry: str(entry["sample_id"]))
    return summary, split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-groups", type=int, default=200)
    parser.add_argument("--calibration-groups", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    input_bytes = args.input.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if input_sha256.lower() != args.expected_input_sha256.lower():
        raise SystemExit("input SHA-256 does not match the required pin")
    rows = json.loads(input_bytes)
    summary, split_rows = build_split(
        rows,
        seed=args.seed,
        dev_groups=args.dev_groups,
        calibration_groups=args.calibration_groups,
    )
    summary["input_sha256"] = input_sha256
    summary["input_rows"] = len(rows)
    summary["input_uploader_groups"] = len({str(row["uploader"]) for row in rows})

    args.output.mkdir(parents=True)
    membership_path = args.output / "split-membership.jsonl"
    membership_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows),
        encoding="utf-8",
    )
    summary["membership_sha256"] = hashlib.sha256(membership_path.read_bytes()).hexdigest()
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
