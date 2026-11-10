"""Inventory prior Perception Test media/feature IDs without opening media bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VIDEO_ID_PATTERN = re.compile(r"\bvideo[_-](\d+)\b", re.IGNORECASE)
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".txt"}
SEALED_PATH_COMPONENTS = {"test", "test_split", "official_test", "sealed_audit", "sealed-audit"}
DIRECT_ARTIFACT_SUFFIXES = {".mp4", ".wav", ".mp3", ".npy", ".npz", ".pt", ".safetensors"}
IGNORED_ARTIFACT_DIRECTORY_PREFIXES = ("perception-test-mcqa-exposure-audit-",)


def _is_direct_artifact_path(value: str) -> bool:
    """Identify paths that indicate media, computed features, or scored cohorts.

    This deliberately does not treat broad source/cut-map manifests as proof of
    model exposure. The raw inventory remains available for manual review.
    """
    normalized = value.lower().replace("\\", "/")
    parts = normalized.split("/")
    name = parts[-1]
    return (
        any(part == "media" or part.startswith("features") for part in parts)
        or Path(name).suffix in DIRECT_ARTIFACT_SUFFIXES
        or "prediction" in name
        or name == "fixed_av_cohort_v1_seed17.json"
        or name
        in {
            "validation_transfer_records.jsonl",
            "valid-routing.json",
            "selected-media.json",
        }
    )


def _video_ids(value: str) -> set[str]:
    return {f"video_{match.group(1)}" for match in VIDEO_ID_PATTERN.finditer(value)}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_question_content(question: Any) -> str:
    if not isinstance(question, dict):
        raise ValueError("question row must be an object")
    prompt = question.get("question")
    options = question.get("options")
    if not isinstance(prompt, str) or not isinstance(options, list):
        raise ValueError("question text and options are required for overlap checks")
    if len(options) != 3 or any(not isinstance(option, str) for option in options):
        raise ValueError("exactly three string options are required for overlap checks")
    def normalize(text: str) -> str:
        return re.sub(
            r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()
        ).strip()
    content = json.dumps(
        [normalize(prompt), sorted(normalize(option) for option in options)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_json_ids(path: Path, max_text_bytes: int) -> tuple[set[str], str | None]:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > max_text_bytes:
        return set(), None
    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    found = _video_ids(text)
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        found.update(_video_ids_from_fields(value))
    return found, _sha256_bytes(payload)


def _video_ids_from_fields(value: Any, field: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"video_id", "video_ids", "source_video_id"}:
                if isinstance(child, int) and not isinstance(child, bool):
                    found.add(f"video_{child}")
                elif isinstance(child, str):
                    found.update(_video_ids(child))
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, int) and not isinstance(item, bool):
                            found.add(f"video_{item}")
                        elif isinstance(item, str):
                            found.update(_video_ids(item))
            found.update(_video_ids_from_fields(child, normalized_key))
    elif isinstance(value, list):
        for child in value:
            found.update(_video_ids_from_fields(child, field))
    return found


def scan_artifact_roots(
    artifact_roots: list[str | Path], *, max_text_bytes: int = 10_000_000
) -> dict[str, Any]:
    """Collect conservative video-ID evidence from paths and small text manifests.

    Media/model/feature payload bytes are never opened. Every Perception-prefixed
    child directory is scanned so older experiment names are not silently missed.
    """
    evidence: dict[str, set[str]] = defaultdict(set)
    root_summaries: list[dict[str, Any]] = []
    text_manifest_records: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    extension_counts: Counter[str] = Counter()

    for root_value in artifact_roots:
        root = Path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(f"artifact root is not a directory: {root}")
        candidate_dirs = sorted(
            item for item in root.iterdir()
            if item.is_dir()
            and item.name.lower().startswith("perception")
            and not item.name.lower().startswith(IGNORED_ARTIFACT_DIRECTORY_PREFIXES)
        )
        root_id_sets: dict[str, set[str]] = defaultdict(set)
        files_seen = 0
        for candidate in candidate_dirs:
            for path in candidate.rglob("*"):
                if not path.is_file():
                    continue
                relative_parts = {part.lower() for part in path.relative_to(candidate).parts}
                if relative_parts & SEALED_PATH_COMPONENTS:
                    continue
                files_seen += 1
                size = path.stat().st_size
                relative = path.relative_to(root).as_posix()
                ids = _video_ids(relative)
                if ids:
                    extension_counts[path.suffix.lower() or "[no extension]"] += 1
                    path_records.append({"path": relative, "bytes": size})
                    for video_id in ids:
                        evidence[video_id].add(relative)
                        root_id_sets[candidate.name].add(video_id)

                content_ids, content_sha = _read_json_ids(path, max_text_bytes)
                if content_sha is not None and content_ids:
                    text_manifest_records.append(
                        {"path": relative, "bytes": size, "sha256": content_sha}
                    )
                    for video_id in content_ids:
                        evidence[video_id].add(relative)
                        root_id_sets[candidate.name].add(video_id)

        root_summaries.append(
            {
                "root": str(root),
                "candidate_directories": len(candidate_dirs),
                "files_seen_by_name_only_or_small_text_scan": files_seen,
                "per_directory_video_ids": {
                    name: len(ids) for name, ids in sorted(root_id_sets.items())
                },
                "unique_video_ids": len(set().union(*root_id_sets.values()))
                if root_id_sets else 0,
            }
        )

    return {
        "artifact_roots": root_summaries,
        "unique_video_ids": sorted(evidence),
        "evidence_by_video_id": {
            video_id: sorted(paths) for video_id, paths in sorted(evidence.items())
        },
        "path_records": sorted(path_records, key=lambda row: row["path"]),
        "text_manifest_records": sorted(text_manifest_records, key=lambda row: row["path"]),
        "path_extension_counts": dict(sorted(extension_counts.items())),
        "media_and_model_payload_bytes_opened": False,
        "limits": [
            "ID presence is conservative prior artifact exposure, not proof that a model "
            "scored the item.",
            "Only supplied roots and Perception-prefixed child directories were scanned.",
            "Video IDs in aggregate text manifests are counted even when the manifest does "
            "not prove model use.",
        ],
    }


def load_question_video_groups(annotation_zip: str | Path, *, split: str) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("only train and valid annotation archives are supported")
    member = f"all_{split}.json"
    with zipfile.ZipFile(annotation_zip) as archive:
        annotations = json.loads(archive.read(member))
    groups: dict[str, int] = {}
    content_hashes_by_video: dict[str, list[str]] = {}
    reasoning_by_video: dict[str, list[str]] = {}
    for video_id, row in annotations.items():
        metadata = row.get("metadata", {})
        if metadata.get("split") != split or metadata.get("video_id") != video_id:
            raise ValueError(f"annotation metadata mismatch for {video_id}")
        questions = row.get("mc_question")
        if not isinstance(questions, list):
            raise ValueError(f"missing MCQA list for {video_id}")
        groups[video_id] = len(questions)
        content_hashes_by_video[video_id] = [
            _normalized_question_content(question) for question in questions
        ]
        reasoning_by_video[video_id] = [
            str(question.get("reasoning", "unknown")) for question in questions
        ]
    return {
        "video_ids": sorted(groups),
        "question_count": sum(groups.values()),
        "questions_by_video": groups,
        "normalized_content_hashes_by_video": content_hashes_by_video,
        "reasoning_by_video": reasoning_by_video,
        "annotation_archive_sha256": sha256_file(annotation_zip),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_exposure_report(
    train_groups: dict[str, Any],
    valid_groups: dict[str, Any],
    artifact_scan: dict[str, Any],
) -> dict[str, Any]:
    train_ids = set(train_groups["video_ids"])
    valid_ids = set(valid_groups["video_ids"])
    if train_ids & valid_ids:
        raise ValueError("official train/valid video IDs overlap")
    touched = set(artifact_scan["unique_video_ids"])
    train_unexposed = sorted(train_ids - touched)
    valid_unexposed = sorted(valid_ids - touched)
    direct_evidence = {
        video_id
        for video_id, paths in artifact_scan["evidence_by_video_id"].items()
        if any(_is_direct_artifact_path(path) for path in paths)
    }
    candidate_valid_ids = sorted(valid_ids - direct_evidence)

    def counts(ids: list[str], groups: dict[str, Any]) -> dict[str, int]:
        return {
            "video_groups": len(ids),
            "questions": sum(groups["questions_by_video"][video_id] for video_id in ids),
        }

    candidate_question_count = sum(
        valid_groups["questions_by_video"][video_id] for video_id in candidate_valid_ids
    )
    train_content = {
        content_hash
        for hashes in train_groups["normalized_content_hashes_by_video"].values()
        for content_hash in hashes
    }
    candidate_novel_questions = {
        video_id: [
            content_hash
            for content_hash in valid_groups["normalized_content_hashes_by_video"][video_id]
            if content_hash not in train_content
        ]
        for video_id in candidate_valid_ids
    }
    novel_question_count = sum(len(hashes) for hashes in candidate_novel_questions.values())
    novel_candidate_video_ids = sorted(
        video_id for video_id, hashes in candidate_novel_questions.items() if hashes
    )
    question_types: dict[str, dict[str, Any]] = {}
    for video_id in candidate_valid_ids:
        content_hashes = valid_groups["normalized_content_hashes_by_video"][video_id]
        reasoning_labels = valid_groups["reasoning_by_video"][video_id]
        for content_hash, reasoning in zip(content_hashes, reasoning_labels, strict=True):
            if content_hash in train_content:
                continue
            entry = question_types.setdefault(
                reasoning, {"questions": 0, "video_ids": set()}
            )
            entry["questions"] += 1
            entry["video_ids"].add(video_id)
    question_type_summary = {
        reasoning: {
            "questions": values["questions"],
            "video_groups": len(values["video_ids"]),
        }
        for reasoning, values in sorted(question_types.items())
    }

    return {
        "protocol": "perception_test_mcqa_conservative_video_exposure_audit_v1",
        "official_train": {
            "video_groups": len(train_ids),
            "questions": train_groups["question_count"],
            "annotation_archive_sha256": train_groups["annotation_archive_sha256"],
        },
        "official_valid": {
            "video_groups": len(valid_ids),
            "questions": valid_groups["question_count"],
            "annotation_archive_sha256": valid_groups["annotation_archive_sha256"],
        },
        "official_train_valid_video_id_overlap": 0,
        "prior_artifact_video_ids": len(touched),
        "prior_artifact_video_id_overlap": {
            "train_video_groups": len(train_ids & touched),
            "valid_video_groups": len(valid_ids & touched),
        },
        "unexposed_by_artifact_inventory": {
            "train": counts(train_unexposed, train_groups),
            "valid": counts(valid_unexposed, valid_groups),
        },
        "unexposed_video_ids": {"train": train_unexposed, "valid": valid_unexposed},
        "direct_artifact_video_id_overlap": {
            "train_video_groups": len(train_ids & direct_evidence),
            "valid_video_groups": len(valid_ids & direct_evidence),
        },
        "validation_candidates_without_direct_artifact_match": {
            "video_groups": len(candidate_valid_ids),
            "questions": candidate_question_count,
            "video_ids": candidate_valid_ids,
            "interpretation": (
                "Candidate pool only: absence of a matching local "
                "media/feature/prediction/cohort path "
                "is not proof of prior non-exposure, semantic novelty, or split suitability."
            ),
        },
        "validation_candidates_after_train_content_deduplication": {
            "video_groups_with_at_least_one_question": len(novel_candidate_video_ids),
            "questions": novel_question_count,
            "question_types": question_type_summary,
            "cross_split_normalized_content_overlap_removed": (
                candidate_question_count - novel_question_count
            ),
            "video_ids": novel_candidate_video_ids,
            "normalization": (
                "Unicode NFKC, casefold, collapsed whitespace, order-insensitive options; "
                "SHA-256 of question plus three options, excluding answer labels."
            ),
            "interpretation": (
                "Question-level exact normalized-content deduplication within validation "
                "candidate videos; video groups remain assigned wholly to validation. "
                "This is not semantic or template-level deduplication."
            ),
        },
        "artifact_scan": artifact_scan,
        "interpretation": (
            "This is an exposure inventory, not a blind split or proof of semantic novelty. "
            "The broad inventory quarantines any artifact mention. A second, path-classified "
            "candidate pool excludes direct media, feature, prediction, and selected-cohort "
            "artifacts, while retaining metadata-only mentions for manual review. No inference "
            "or fitting occurs."
        ),
        "next_gate": (
            "Do not score or fit until a separate protocol freezes a video-grouped "
            "train/validation "
            "cohort, option/task controls, and cross-source content checks."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--valid-annotations", required=True)
    parser.add_argument("--artifact-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    train = load_question_video_groups(args.train_annotations, split="train")
    valid = load_question_video_groups(args.valid_annotations, split="valid")
    scan = scan_artifact_roots(args.artifact_root)
    report = build_exposure_report(train, valid, scan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "artifact_scan"}
    for key in (
        "unexposed_video_ids",
        "validation_candidates_without_direct_artifact_match",
        "validation_candidates_after_train_content_deduplication",
    ):
        if key in summary and isinstance(summary[key], dict):
            summary[key].pop("video_ids", None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
