"""Fetch a tiny deterministic CLEVRER clip sample using bounded ZIP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any

import requests

from tiny_omni_decision.clevrer import TAXONOMIES, classify

TRAIN_URL = "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip"
TRAIN_BYTES = 12_354_893_389
TRAIN_ETAG = '"2e068b64d-59db378b1c46e"'
VALIDATION_URL = "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"
VALIDATION_BYTES = 6_206_059_321
VALIDATION_ETAG = '"171e8f339-59db35a74239f"'
TRAIN_QUESTIONS_SHA256 = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
VALIDATION_QUESTIONS_SHA256 = "fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d"
REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
PREVIOUSLY_USED_SCENES = {"train": {0, 1, 2}, "validation": {10_000, 10_001, 10_002}}
RANGE_READ_AHEAD_BYTES = 1024 * 1024
DEFAULT_DOWNLOAD_WORKERS = 4


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RangeFile(io.RawIOBase):
    """Seekable HTTP range file with immutable size/ETag checks."""

    def __init__(self, url: str, expected_bytes: int, expected_etag: str):
        self.url = url
        self.size = expected_bytes
        self.etag = expected_etag
        self.position = 0
        self.bytes_read = 0
        self.range_request_count = 0
        self.buffer_start = 0
        self.buffer = b""
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        if int(head.headers.get("Content-Length", -1)) != expected_bytes:
            raise ValueError("CLEVRER video archive size changed")
        if head.headers.get("ETag") != expected_etag:
            raise ValueError("CLEVRER video archive ETag changed")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        target = (
            offset
            if whence == io.SEEK_SET
            else self.position + offset
            if whence == io.SEEK_CUR
            else self.size + offset
        )
        if target < 0:
            raise ValueError("negative archive seek")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.position >= self.size:
            return b""
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        buffer_offset = self.position - self.buffer_start
        if 0 <= buffer_offset < len(self.buffer):
            available = min(count, len(self.buffer) - buffer_offset)
            data = self.buffer[buffer_offset : buffer_offset + available]
            self.position += len(data)
            return data

        fetch_count = max(count, RANGE_READ_AHEAD_BYTES)
        end = min(self.position + fetch_count - 1, self.size - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"},
            timeout=120,
        )
        response.raise_for_status()
        if (
            response.status_code != 206
            or response.headers.get("Content-Range", "")
            != f"bytes {self.position}-{end}/{self.size}"
        ):
            raise RuntimeError("CLEVRER video archive did not honor the requested byte range")
        if response.headers.get("ETag") != self.etag:
            raise RuntimeError("CLEVRER video archive ETag changed during range reads")
        data = response.content
        if len(data) != count:
            if len(data) != end - self.position + 1:
                raise RuntimeError("CLEVRER range response size mismatch")
        self.buffer_start = self.position
        self.buffer = data
        returned = data[:count]
        self.position += len(returned)
        self.bytes_read += len(data)
        self.range_request_count += 1
        return returned


def choose(
    rows: list[dict[str, Any]],
    split: str,
    per_type: int,
    seed: int,
    excluded_content: set[tuple[str, tuple[str, ...]]] | None = None,
    excluded_scenes: set[int] | None = None,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, list[dict[str, Any]]]] = {}
    used = PREVIOUSLY_USED_SCENES[split] | (excluded_scenes or set())
    for row in rows:
        scene = int(row["scene_index"])
        if scene in used:
            continue
        for question in row.get("questions", []):
            if question.get("question_type") != "descriptive":
                continue
            taxonomy = question.get("question_subtype")
            target = question.get("answer")
            if taxonomy not in TAXONOMIES or target not in TAXONOMIES[taxonomy]:
                continue
            content = (
                str(question["question"]).casefold().strip(),
                tuple(sorted(TAXONOMIES[taxonomy])),
            )
            if excluded_content and content in excluded_content:
                continue
            task = classify(question)
            candidates.setdefault(scene, {"temporal_descriptive": [], "static_descriptive": []})[
                task
            ].append(
                {
                    "id": f"clevrer-{split}-{scene:05d}-q{int(question['question_id']):03d}",
                    "split": split,
                    "scene_index": scene,
                    "video_filename": row["video_filename"],
                    "question_id": int(question["question_id"]),
                    "question_type": task,
                    "taxonomy": taxonomy,
                    "question": str(question["question"]),
                    "options": TAXONOMIES[taxonomy],
                    "target": str(target),
                    "program": question.get("program", []),
                }
            )
    eligible = [
        scene
        for scene, by_type in candidates.items()
        if by_type["temporal_descriptive"] and by_type["static_descriptive"]
    ]
    eligible.sort(
        key=lambda scene: hashlib.sha256(f"{seed}|{split}|scene|{scene}".encode()).hexdigest()
    )
    if len(eligible) < per_type:
        raise ValueError(
            f"{split} has only {len(eligible)} unused scenes with both question types; "
            f"need {per_type}"
        )
    selected = []
    for scene in eligible[:per_type]:
        for task in ("temporal_descriptive", "static_descriptive"):
            choices = candidates[scene][task]
            choices.sort(
                key=lambda item: hashlib.sha256(
                    f"{seed}|{split}|{scene}|{task}|{item['question_id']}".encode()
                ).hexdigest()
            )
            selected.append(choices[0])
    return sorted(selected, key=lambda item: (item["scene_index"], item["question_id"]))


def choose_validation(
    rows: list[dict[str, Any]],
    per_type: int,
    seed: int,
    train_content: set[tuple[str, tuple[str, ...]]],
    excluded_scenes: set[int],
    allow_question_template_overlap: bool = False,
) -> list[dict[str, Any]]:
    return choose(
        rows,
        "validation",
        per_type,
        seed,
        excluded_content=None if allow_question_template_overlap else train_content,
        excluded_scenes=excluded_scenes,
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-questions", type=Path, required=True)
    parser.add_argument("--validation-questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-type", type=int, default=16)
    parser.add_argument("--validation-per-type", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="reuse only existing selected MP4s that match pinned ZIP size and CRC",
    )
    parser.add_argument(
        "--exclude-scenes-json",
        type=Path,
        help="optional JSON mapping train/validation to excluded scene IDs",
    )
    parser.add_argument(
        "--allow-question-template-overlap",
        action="store_true",
        help=(
            "allow normalized question templates across scene-disjoint splits; "
            "the report counts shared templates"
        ),
    )
    return parser.parse_args()


def _extract_archive_chunk(
    records: list[dict[str, Any]],
    source: dict[str, Any],
    output: Path,
    split: str,
    *,
    resume_partial: bool = False,
) -> tuple[dict[int, tuple[str, int, str]], int, int, int]:
    archive_file = RangeFile(source["url"], source["bytes"], source["etag"])
    extracted: dict[int, tuple[str, int, str]] = {}
    reused_media = 0
    with zipfile.ZipFile(archive_file) as archive:
        names_by_basename: dict[str, list[str]] = {}
        for name in archive.namelist():
            if name.lower().endswith(".mp4"):
                names_by_basename.setdefault(Path(name).name, []).append(name)
        for record in records:
            matches = names_by_basename.get(record["video_filename"], [])
            if len(matches) != 1:
                raise ValueError(
                    "expected one archive member for "
                    f"{record['video_filename']}, found {len(matches)}"
                )
            member = matches[0]
            media_dir = output / "media" / split
            media_dir.mkdir(parents=True, exist_ok=True)
            path = media_dir / record["video_filename"]
            if path.exists():
                if not resume_partial:
                    raise FileExistsError(
                        f"existing selected media requires --resume-partial: {path}"
                    )
                info = archive.getinfo(member)
                media = path.read_bytes()
                if (
                    len(media) != info.file_size
                    or zlib.crc32(media) & 0xFFFFFFFF != info.CRC
                ):
                    raise ValueError(f"partial media does not match pinned ZIP entry: {path}")
                reused_media += 1
            else:
                media = archive.read(member)
            if len(media) < 16 or b"ftyp" not in media[:16]:
                raise ValueError(f"invalid MP4 media bytes for {member}")
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(media)
            extracted[int(record["scene_index"])] = (member, len(media), sha256(media))
    return extracted, archive_file.bytes_read, archive_file.range_request_count, reused_media


def _merge_archive_worker_results(
    results: list[tuple[dict[int, tuple[str, int, str]], int, int, int]],
) -> dict[str, Any]:
    media_by_scene: dict[int, tuple[str, int, str]] = {}
    for extracted, _, _, _ in results:
        duplicate_scenes = set(media_by_scene) & set(extracted)
        if duplicate_scenes:
            raise ValueError(f"archive workers extracted duplicate scenes: {duplicate_scenes}")
        media_by_scene.update(extracted)
    return {
        "media_by_scene": media_by_scene,
        "range_bytes_read": sum(result[1] for result in results),
        "range_request_count": sum(result[2] for result in results),
        "reused_media_count": sum(result[3] for result in results),
    }


def _write_integrity_receipt(
    output: Path,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    *,
    allow_question_template_overlap: bool,
) -> Path:
    scene_media: dict[str, dict[str, str]] = {"train": {}, "validation": {}}
    media_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    content: dict[str, set[tuple[str, tuple[str, ...]]]] = {"train": set(), "validation": set()}
    verified_media: set[str] = set()
    for split, rows in (("train", train), ("validation", validation)):
        for row in rows:
            if row["split"] != split or row["source"] != f"MIT-IBM/CLEVRER@{REVISION}":
                raise ValueError(f"incorrect split/source identity for {row['id']}")
            sample_id = str(row["id"])
            if sample_id in ids[split]:
                raise ValueError(f"duplicate sample id: {sample_id}")
            ids[split].add(sample_id)
            scene = str(row["scene_group_id"])
            media_hash = str(row["media_sha256"])
            if scene in scene_media[split] and scene_media[split][scene] != media_hash:
                raise ValueError(f"scene group maps to multiple media assets: {scene}")
            scene_media[split][scene] = media_hash
            media_hashes[split].add(media_hash)
            path = (output / str(row["media_path"])).resolve(strict=True)
            try:
                path.relative_to(output.resolve(strict=True))
            except ValueError as error:
                raise ValueError(f"media path escapes frozen sample directory: {path}") from error
            if media_hash not in verified_media:
                if sha256(path.read_bytes()) != media_hash:
                    raise ValueError(f"post-write media hash mismatch for {sample_id}")
                verified_media.add(media_hash)
            content[split].add(
                (str(row["question"]).casefold().strip(), tuple(sorted(row["options"])))
            )
    scene_overlap = set(scene_media["train"]) & set(scene_media["validation"])
    media_overlap = media_hashes["train"] & media_hashes["validation"]
    content_overlap = content["train"] & content["validation"]
    sample_id_overlap = ids["train"] & ids["validation"]
    if scene_overlap or media_overlap or sample_id_overlap:
        raise ValueError("train/validation scene or media identity overlap")
    if content_overlap and not allow_question_template_overlap:
        raise ValueError("normalized question/options overlap across splits")
    gate = {
        "status": "passed",
        "all_media_sha256_verified": True,
        "group_integrity": {
            "train_validation_scene_overlap": len(scene_overlap),
            "train_validation_media_overlap": len(media_overlap),
            "train_unique_scenes": len(scene_media["train"]),
            "validation_unique_scenes": len(scene_media["validation"]),
        },
        "sample_hashes": {
            "train_jsonl": hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
            "validation_jsonl": hashlib.sha256(
                (output / "validation.jsonl").read_bytes()
            ).hexdigest(),
        },
        "sample_id_overlap": len(sample_id_overlap),
        "normalized_question_option_overlap": len(content_overlap),
        "train_unique_media": len(media_hashes["train"]),
        "validation_unique_media": len(media_hashes["validation"]),
        "sealed_audit_loaded": False,
    }
    receipt = output / "media-integrity-v2.json"
    receipt.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume_partial:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    if args.train_per_type < 1 or args.validation_per_type < 1:
        raise ValueError("per-type sample sizes must be positive")
    train_bytes = args.train_questions.read_bytes()
    val_bytes = args.validation_questions.read_bytes()
    if (
        sha256(train_bytes) != TRAIN_QUESTIONS_SHA256
        or sha256(val_bytes) != VALIDATION_QUESTIONS_SHA256
    ):
        raise ValueError("CLEVRER question JSON hash does not match pinned source revision")
    train_rows = json.loads(train_bytes)
    val_rows = json.loads(val_bytes)
    excluded_scenes = {"train": set(), "validation": set()}
    exclude_scenes_sha256 = None
    if args.exclude_scenes_json:
        excluded_bytes = args.exclude_scenes_json.read_bytes()
        excluded_raw = json.loads(excluded_bytes)
        if not isinstance(excluded_raw, dict) or set(excluded_raw) != set(excluded_scenes):
            raise ValueError("excluded-scene JSON must have exactly train and validation keys")
        for split, raw_scenes in excluded_raw.items():
            if not isinstance(raw_scenes, list) or any(
                not isinstance(scene, int) or isinstance(scene, bool) or scene < 0
                for scene in raw_scenes
            ):
                raise ValueError(f"excluded {split} scene IDs must be non-negative integers")
            if len(raw_scenes) != len(set(raw_scenes)):
                raise ValueError(f"excluded {split} scene IDs must be unique")
            excluded_scenes[split] = set(raw_scenes)
        exclude_scenes_sha256 = sha256(excluded_bytes)
    train = choose(
        train_rows,
        "train",
        args.train_per_type,
        args.seed,
        excluded_scenes=excluded_scenes["train"],
    )
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    validation = choose_validation(
        val_rows,
        args.validation_per_type,
        args.seed,
        train_content,
        excluded_scenes["validation"],
        allow_question_template_overlap=args.allow_question_template_overlap,
    )
    validation_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in validation
    }
    question_content_overlap = train_content & validation_content
    if question_content_overlap and not args.allow_question_template_overlap:
        raise ValueError("normalized CLEVRER question content overlaps across splits")
    train_scenes = {item["scene_index"] for item in train}
    val_scenes = {item["scene_index"] for item in validation}
    if train_scenes & val_scenes:
        raise ValueError("CLEVRER scene overlap across train and validation")
    expected_media_paths = {
        Path("media") / split / record["video_filename"]
        for split, records in (("train", train), ("validation", validation))
        for record in records
    }
    if args.resume_partial and output.exists():
        existing_files = {
            path.relative_to(output)
            for path in output.rglob("*")
            if path.is_file()
        }
        unexpected_files = existing_files - expected_media_paths
        if unexpected_files:
            raise ValueError(
                "partial output contains non-media or unselected files: "
                f"{sorted(map(str, unexpected_files))[:5]}"
            )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    archive_sources = {
        "train": {"url": TRAIN_URL, "bytes": TRAIN_BYTES, "etag": TRAIN_ETAG},
        "validation": {
            "url": VALIDATION_URL,
            "bytes": VALIDATION_BYTES,
            "etag": VALIDATION_ETAG,
        },
    }
    archive_bytes_read = {}
    archive_range_request_count = {}
    resumed_media_count = {}
    if args.download_workers < 1 or args.download_workers > 8:
        raise ValueError("download-workers must be between 1 and 8")
    for split, records in (("train", train), ("validation", validation)):
        source = archive_sources[split]
        unique_records = {
            int(record["scene_index"]): record
            for record in records
        }
        worker_count = min(args.download_workers, len(unique_records))
        chunks = [
            list(unique_records.values())[offset::worker_count]
            for offset in range(worker_count)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _extract_archive_chunk,
                    chunk,
                    source,
                    output,
                    split,
                    resume_partial=args.resume_partial,
                )
                for chunk in chunks
            ]
            results = [future.result() for future in futures]
        merged_results = _merge_archive_worker_results(results)
        extracted_by_scene = merged_results["media_by_scene"]
        archive_bytes_read[split] = merged_results["range_bytes_read"]
        archive_range_request_count[split] = merged_results["range_request_count"]
        resumed_media_count[split] = merged_results["reused_media_count"]
        for record in records:
            member, media_size, media_hash = extracted_by_scene[int(record["scene_index"])]
            path = output / "media" / split / record["video_filename"]
            record.update(
                {
                    "source": f"MIT-IBM/CLEVRER@{REVISION}",
                    "scene_group_id": f"clevrer:{split}:{record['scene_index']}",
                    "media_path": path.relative_to(output).as_posix(),
                    "media_bytes": media_size,
                    "media_sha256": media_hash,
                    "archive_member": member,
                }
            )
    all_hashes = {
        split: {item["media_sha256"] for item in records}
        for split, records in (("train", train), ("validation", validation))
    }
    if all_hashes["train"] & all_hashes["validation"]:
        raise ValueError("CLEVRER media content overlap across train and validation")
    for split, records in (("train", train), ("validation", validation)):
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "status": "complete_video_sample",
        "dataset": "MIT-IBM/CLEVRER",
        "revision": REVISION,
        "question_hashes": {
            "train": TRAIN_QUESTIONS_SHA256,
            "validation": VALIDATION_QUESTIONS_SHA256,
        },
        "seed": args.seed,
        "download_workers": args.download_workers,
        "resumed_media_count": resumed_media_count,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_question_type_counts": {
            kind: sum(row["question_type"] == kind for row in train)
            for kind in ("temporal_descriptive", "static_descriptive")
        },
        "validation_question_type_counts": {
            kind: sum(row["question_type"] == kind for row in validation)
            for kind in ("temporal_descriptive", "static_descriptive")
        },
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(val_scenes),
        "scene_overlap": 0,
        "normalized_question_content_overlap": len(question_content_overlap),
        "validation_rows_with_shared_question_template": sum(
            (
                str(row["question"]).casefold().strip(),
                tuple(sorted(row["options"])),
            ) in train_content
            for row in validation
        ),
        "question_template_overlap_allowed": args.allow_question_template_overlap,
        "media_hash_overlap": 0,
        "previous_scenes_excluded": {
            split: sorted(PREVIOUSLY_USED_SCENES[split] | excluded_scenes[split])
            for split in PREVIOUSLY_USED_SCENES
        },
        "exclude_scenes_json_sha256": exclude_scenes_sha256,
        "archive_sources": {
            split: {
                "size_bytes": source["bytes"],
                "etag": source["etag"],
                "range_bytes_read": archive_bytes_read[split],
                "range_request_count": archive_range_request_count[split],
            }
            for split, source in archive_sources.items()
        },
        "archive_sha256_verified": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "selected media hashes are verified; SHA-256 is not computed over either full "
            "video archive"
        ] + ([
            "normalized question templates repeat across train and validation; "
            "scene and media identities remain disjoint"
        ] if question_content_overlap else []),
    }
    fetch_report_path = output / "fetch-report.json"
    fetch_report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    integrity_path = _write_integrity_receipt(
        output,
        train,
        validation,
        allow_question_template_overlap=args.allow_question_template_overlap,
    )
    report["integrity_report_sha256"] = sha256(integrity_path.read_bytes())
    fetch_report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
