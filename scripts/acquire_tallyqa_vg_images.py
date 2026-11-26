"""Acquire only the fixed TallyQA VG AMT train/validation image set."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

SOURCE_COMMIT = "46cdc649ec79c3dcc2720ff227ad07d7ee51da6f"
ARCHIVE_SHA256 = "e805199fa25c91d2f9df964920c6c36ce8d6cb941c970176f38a92f971bab5ea"
ALLOWED_PREFIXES = frozenset({"VG_100K", "VG_100K_2"})
VG_BASE_URL = "https://cs.stanford.edu/people/rak248"
MAX_IMAGE_BYTES = 25_000_000
MAX_WORKERS = 4


@dataclass(frozen=True)
class ImageJob:
    split: str
    image_id: str
    source_path: str


@dataclass(frozen=True)
class ImageReceipt:
    split: str
    image_id: str
    source_path: str
    url: str
    http_status: int
    content_type: str
    bytes: int
    sha256: str
    width: int
    height: int
    output_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_jobs(screen: dict[str, Any]) -> list[ImageJob]:
    if screen.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("screen manifest does not use the pinned TallyQA commit")
    if screen.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("screen manifest does not use the pinned annotation archive")
    split = screen.get("split")
    if not isinstance(split, dict) or split.get("image_overlap_count") != 0:
        raise ValueError("screen manifest lacks a verified image-disjoint split")
    if split.get("normalized_question_overlap_count") != 0:
        raise ValueError("screen manifest has normalized-question overlap")

    jobs: list[ImageJob] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for split_name in ("train", "validation"):
        entry = split.get(split_name)
        if not isinstance(entry, dict) or not isinstance(entry.get("image_paths"), dict):
            raise ValueError(f"missing {split_name} image path manifest")
        image_ids = [str(value) for value in entry.get("image_ids", [])]
        path_map = {str(key): str(value) for key, value in entry["image_paths"].items()}
        if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(path_map):
            raise ValueError(f"{split_name} image IDs and path map disagree")
        for image_id in image_ids:
            source_path = path_map[image_id].replace("\\", "/")
            parsed = PurePosixPath(source_path)
            if (
                parsed.is_absolute()
                or len(parsed.parts) != 2
                or parsed.parts[0] not in ALLOWED_PREFIXES
                or parsed.suffix.lower() not in {".jpg", ".jpeg"}
                or ".." in parsed.parts
            ):
                raise ValueError(f"disallowed Visual Genome asset path: {source_path}")
            if image_id in seen_ids or source_path in seen_paths:
                raise ValueError("duplicate image identity/path across selected splits")
            seen_ids.add(image_id)
            seen_paths.add(source_path)
            jobs.append(ImageJob(split_name, image_id, source_path))
    return jobs


def _download_one(job: ImageJob, output_dir: Path) -> ImageReceipt:
    from PIL import Image

    target = output_dir / job.split / f"{job.image_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() or part.exists():
        raise FileExistsError(f"refusing to overwrite or reuse unreceipted output: {target}")

    url = f"{VG_BASE_URL}/{quote(job.source_path, safe='/')}"
    request = Request(url, headers={"User-Agent": "tiny-omni-decision/1.0", "Accept": "image/jpeg"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=45) as response:
                status = int(response.status)
                content_type = response.headers.get_content_type()
                if status != 200 or content_type != "image/jpeg":
                    raise ValueError(f"unexpected image response: {status} {content_type}")
                content_length = response.headers.get("Content-Length")
                written = 0
                digest = hashlib.sha256()
                with part.open("wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > MAX_IMAGE_BYTES:
                            raise ValueError("image exceeds the fixed per-file size guard")
                        digest.update(block)
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if content_length and written != int(content_length):
                raise ValueError("downloaded byte count does not match Content-Length")
            with part.open("rb") as stream:
                if stream.read(2) != b"\xff\xd8":
                    raise ValueError("response is not a JPEG file")
            with Image.open(part) as image:
                image.verify()
            with Image.open(part) as image:
                width, height = image.size
                if image.format != "JPEG" or width < 1 or height < 1:
                    raise ValueError("decoded image metadata is invalid")
            os.replace(part, target)
            return ImageReceipt(
                split=job.split,
                image_id=job.image_id,
                source_path=job.source_path,
                url=url,
                http_status=status,
                content_type=content_type,
                bytes=written,
                sha256=digest.hexdigest(),
                width=width,
                height=height,
                output_path=str(target),
            )
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if part.exists():
                raise RuntimeError(f"partial download retained for inspection: {part}") from error
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"image download failed for {job.source_path}: {last_error}")


def acquire(screen_path: Path, output_dir: Path, workers: int = MAX_WORKERS) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    jobs = _image_jobs(screen)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    receipts: list[ImageReceipt] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_download_one, job, output_dir): job for job in jobs}
        completed = 0
        for future in concurrent.futures.as_completed(pending):
            job = pending[future]
            completed += 1
            try:
                receipts.append(future.result())
            except Exception as error:  # persist exact failure evidence
                failures.append(
                    {"split": job.split, "image_id": job.image_id, "error": repr(error)}
                )
            if completed % 100 == 0 or completed == len(jobs):
                print(f"completed {completed}/{len(jobs)} selected images", flush=True)

    receipts.sort(key=lambda item: (item.split, int(item.image_id)))
    train_hashes = {item.sha256 for item in receipts if item.split == "train"}
    validation_hashes = {item.sha256 for item in receipts if item.split == "validation"}
    content_overlap = train_hashes & validation_hashes
    status = (
        "complete"
        if not failures and not content_overlap and len(receipts) == len(jobs)
        else "incomplete"
    )
    result = {
        "status": status,
        "source_manifest_sha256": _sha256(screen_path),
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": ARCHIVE_SHA256,
        "workers": workers,
        "selected_image_count": len(jobs),
        "downloaded_image_count": len(receipts),
        "bytes_total": sum(item.bytes for item in receipts),
        "unique_image_sha256_count": len({item.sha256 for item in receipts}),
        "train_validation_content_hash_overlap_count": len(content_overlap),
        "train_receipts": [asdict(item) for item in receipts if item.split == "train"],
        "validation_receipts": [
            asdict(item) for item in receipts if item.split == "validation"
        ],
        "failures": failures,
    }
    report_name = "acquisition.json" if status == "complete" else "acquisition-incomplete.json"
    (output_dir / report_name).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    result = acquire(args.screen_manifest, args.output_dir, args.workers)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"train_receipts", "validation_receipts"}
            },
            indent=2,
        )
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
