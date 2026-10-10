"""Fetch a tiny deterministic CLEVRER clip sample using bounded ZIP ranges."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

TRAIN_URL = "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip"
TRAIN_BYTES = 12_354_893_389
TRAIN_ETAG = '"2e068b64d-59db378b1c46e"'
VALIDATION_URL = "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"
VALIDATION_BYTES = 6_206_059_321
VALIDATION_ETAG = '"171e8f339-59db35a74239f"'
TRAIN_QUESTIONS_SHA256 = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
VALIDATION_QUESTIONS_SHA256 = "fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d"
REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
TASK_TOKENS = {
    "after",
    "before",
    "end",
    "filter_collision",
    "filter_in",
    "filter_moving",
    "filter_out",
    "get_frame",
    "start",
}
TASK_RE = re.compile(
    r"\b(first|last|before|after|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|begin(?:s|ning)?|ends?|collision|collide|moving|moves?|move|frame)\b",
    re.I,
)
TAXONOMIES = {
    "exist": ["no", "yes"],
    "query_color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "query_material": ["rubber", "metal"],
    "query_shape": ["cube", "sphere", "cylinder"],
    "count": [str(index) for index in range(6)],
}
PREVIOUSLY_USED_SCENES = {"train": {0, 1, 2}, "validation": {10_000, 10_001, 10_002}}


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
        end = self.position + count - 1
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
            raise RuntimeError("CLEVRER range response size mismatch")
        self.position += len(data)
        self.bytes_read += len(data)
        return data


def classify(question: dict[str, Any]) -> str:
    program = question.get("program")
    temporal = (
        bool(TASK_TOKENS.intersection(map(str, program)))
        if isinstance(program, list)
        else bool(TASK_RE.search(str(question.get("question") or "")))
    )
    return "temporal_descriptive" if temporal else "static_descriptive"


def choose(
    rows: list[dict[str, Any]],
    split: str,
    per_type: int,
    seed: int,
    excluded_content: set[tuple[str, tuple[str, ...]]] | None = None,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, list[dict[str, Any]]]] = {}
    used = PREVIOUSLY_USED_SCENES[split]
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-questions", type=Path, required=True)
    parser.add_argument("--validation-questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-type", type=int, default=16)
    parser.add_argument("--validation-per-type", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
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
    train = choose(train_rows, "train", args.train_per_type, args.seed)
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    validation = choose(
        val_rows,
        "validation",
        args.validation_per_type,
        args.seed,
        excluded_content=train_content,
    )
    validation_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in validation
    }
    if train_content & validation_content:
        raise ValueError("normalized CLEVRER question content overlaps across splits")
    train_scenes = {item["scene_index"] for item in train}
    val_scenes = {item["scene_index"] for item in validation}
    if train_scenes & val_scenes:
        raise ValueError("CLEVRER scene overlap across train and validation")
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
    for split, records in (("train", train), ("validation", validation)):
        source = archive_sources[split]
        archive_file = RangeFile(source["url"], source["bytes"], source["etag"])
        with zipfile.ZipFile(archive_file) as archive:
            names_by_basename: dict[str, list[str]] = {}
            for name in archive.namelist():
                if name.lower().endswith(".mp4"):
                    names_by_basename.setdefault(Path(name).name, []).append(name)
            extracted_by_scene: dict[int, tuple[str, int, str]] = {}
            for record in records:
                if record["scene_index"] in extracted_by_scene:
                    member, media_size, media_hash = extracted_by_scene[record["scene_index"]]
                    record.update(
                        {
                            "source": f"MIT-IBM/CLEVRER@{REVISION}",
                            "scene_group_id": f"clevrer:{split}:{record['scene_index']}",
                            "media_path": f"media/{split}/{record['video_filename']}",
                            "media_bytes": media_size,
                            "media_sha256": media_hash,
                            "archive_member": member,
                        }
                    )
                    continue
                matches = names_by_basename.get(record["video_filename"], [])
                if len(matches) != 1:
                    raise ValueError(
                        "expected one archive member for "
                        f"{record['video_filename']}, found {len(matches)}"
                    )
                member = matches[0]
                media = archive.read(member)
                if len(media) < 16 or b"ftyp" not in media[:16]:
                    raise ValueError(f"invalid MP4 media bytes for {member}")
                media_dir = output / "media" / split
                media_dir.mkdir(parents=True, exist_ok=True)
                path = media_dir / record["video_filename"]
                path.write_bytes(media)
                record.update(
                    {
                        "source": f"MIT-IBM/CLEVRER@{REVISION}",
                        "scene_group_id": f"clevrer:{split}:{record['scene_index']}",
                        "media_path": path.relative_to(output).as_posix(),
                        "media_bytes": len(media),
                        "media_sha256": sha256(media),
                        "archive_member": member,
                    }
                )
                extracted_by_scene[record["scene_index"]] = (member, len(media), sha256(media))
        archive_bytes_read[split] = archive_file.bytes_read
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
        "normalized_question_content_overlap": 0,
        "media_hash_overlap": 0,
        "previous_scenes_excluded": {
            split: sorted(scenes) for split, scenes in PREVIOUSLY_USED_SCENES.items()
        },
        "archive_sources": {
            split: {
                "size_bytes": source["bytes"],
                "etag": source["etag"],
                "range_bytes_read": archive_bytes_read[split],
            }
            for split, source in archive_sources.items()
        },
        "archive_sha256_verified": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "selected media hashes are verified; SHA-256 is not computed over either full "
            "video archive"
        ],
    }
    (output / "fetch-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
