"""Fetch a deterministic, split-safe image sample by HTTP byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

ARCHIVE_URL = "https://thor.robots.ox.ac.uk/clevr4/clevr_4_10k_v1.zip"
EXPECTED_ARCHIVE_BYTES = 3_797_490_816
EXPECTED_ARCHIVE_SHA512 = (
    "769465d90b6550a242f6d5940b52f2e0944c65af09e17e6f3f2cb65701af66057"
    "ab06485b2194fa9d8932e03a03e87b3fadece43049e1af1ddd9f8e4990ed19a"
)


class HttpRangeFile(io.RawIOBase):
    """Small seekable file facade over an HTTP range-capable ZIP archive."""

    def __init__(self, url: str = ARCHIVE_URL):
        self.url = url
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers.get("ETag")
        self.position = 0
        self.session = requests.Session()
        if self.size != EXPECTED_ARCHIVE_BYTES:
            raise ValueError(f"unexpected pinned archive size: {self.size}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported seek mode {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.position >= self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"},
            timeout=120,
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"server did not honor byte range: HTTP {response.status_code}")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {self.position}-{end}/"):
            raise RuntimeError(f"unexpected content range: {content_range}")
        if self.etag and response.headers.get("ETag") != self.etag:
            raise RuntimeError("archive ETag changed between HEAD and range response")
        data = response.content
        if len(data) != end - self.position + 1:
            raise RuntimeError("range response has an unexpected byte count")
        self.position += len(data)
        return data


def deterministic_split_sample(
    annotations: dict[str, dict[str, Any]], *, split: str, limit: int, seed: int
) -> list[str]:
    if limit < 1:
        raise ValueError("sample limit must be positive")
    ids = sorted(record_id for record_id, row in annotations.items() if row.get("split") == split)
    if len(ids) < limit:
        raise ValueError(f"split {split!r} has {len(ids)} images, fewer than requested {limit}")
    random.Random(f"{seed}:clevr4:{split}").shuffle(ids)
    return sorted(ids[:limit])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=1024)
    parser.add_argument("--validation-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--labels-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if args.labels_only:
        if not output.is_dir():
            raise FileNotFoundError(f"sample output directory is missing: {output}")
        with zipfile.ZipFile(HttpRangeFile()) as zipped:
            annotation_bytes = zipped.read("clevr_4_annots.json")
        annotations = json.loads(annotation_bytes)
        for split in ("train", "val"):
            manifest_path = output / f"{split}-image-manifest.jsonl"
            rows = [
                json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            label_path = output / f"{split}-labels.jsonl"
            label_path.write_text(
                "".join(
                    json.dumps({"id": row["id"], **annotations[row["id"]]}, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
        report_path = output / "fetch-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["annotation_bytes_sha256"] = sha256_bytes(annotation_bytes)
        report.pop("annotation_sha512_source_archive_member", None)
        report["label_manifest_hashes"] = {
            f"{split}_labels_sha256": hashlib.sha256(
                (output / f"{split}-labels.jsonl").read_bytes()
            ).hexdigest()
            for split in ("train", "val")
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "labels_added",
                    "label_manifest_hashes": report["label_manifest_hashes"],
                },
                indent=2,
            )
        )
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    archive = HttpRangeFile()
    with zipfile.ZipFile(archive) as zipped:
        annotation_bytes = zipped.read("clevr_4_annots.json")
        annotations = json.loads(annotation_bytes)
        selected = {
            "train": deterministic_split_sample(
                annotations, split="train", limit=args.train_images, seed=args.seed
            ),
            "val": deterministic_split_sample(
                annotations, split="val", limit=args.validation_images, seed=args.seed
            ),
        }
    if set(selected["train"]) & set(selected["val"]):
        raise AssertionError("train/validation image IDs overlap")

    def extract_batch(split: str, record_ids: list[str]) -> list[dict[str, Any]]:
        extracted = []
        with zipfile.ZipFile(HttpRangeFile()) as zipped:
            for record_id in record_ids:
                data = zipped.read(f"images/{record_id}.png")
                split_dir = output / split / "images"
                split_dir.mkdir(parents=True, exist_ok=True)
                destination = split_dir / f"{record_id}.png"
                destination.write_bytes(data)
                extracted.append(
                    {
                        "id": record_id,
                        "split": split,
                        "bytes": len(data),
                        "sha256": sha256_bytes(data),
                    }
                )
        return extracted

    results: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split, ids in selected.items():
            for worker_ids in (ids[index :: args.workers] for index in range(args.workers)):
                if worker_ids:
                    futures.append((split, executor.submit(extract_batch, split, worker_ids)))
        for split, future in futures:
            results[split].extend(future.result())

    for split in results:
        results[split].sort(key=lambda row: row["id"])
        (output / f"{split}-image-manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in results[split]),
            encoding="utf-8",
        )
        (output / f"{split}-labels.jsonl").write_text(
            "".join(
                json.dumps({"id": row["id"], **annotations[row["id"]]}, sort_keys=True) + "\n"
                for row in results[split]
            ),
            encoding="utf-8",
        )
    train_ids = {row["id"] for row in results["train"]}
    validation_ids = {row["id"] for row in results["val"]}
    if train_ids & validation_ids:
        raise AssertionError("extracted train/validation image identity overlap")
    report = {
        "schema_version": 1,
        "status": "complete_selected_assets",
        "source_url": ARCHIVE_URL,
        "source_archive_bytes": archive.size,
        "source_archive_sha512_expected_from_manifest": EXPECTED_ARCHIVE_SHA512,
        "source_archive_full_hash_verified": False,
        "source_archive_etag": archive.etag,
        "annotation_bytes_sha256": sha256_bytes(annotation_bytes),
        "seed": args.seed,
        "train_images": len(results["train"]),
        "validation_images": len(results["val"]),
        "train_validation_identity_overlap": 0,
        "train_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in results["train"]).encode()
        ).hexdigest(),
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in results["val"]).encode()
        ).hexdigest(),
        "train_unique_asset_sha256": len({row["sha256"] for row in results["train"]}),
        "validation_unique_asset_sha256": len({row["sha256"] for row in results["val"]}),
        "image_manifest_hashes": {
            f"{split}_image_manifest_sha256": hashlib.sha256(
                (output / f"{split}-image-manifest.jsonl").read_bytes()
            ).hexdigest()
            for split in results
        },
        "label_manifest_hashes": {
            f"{split}_labels_sha256": hashlib.sha256(
                (output / f"{split}-labels.jsonl").read_bytes()
            ).hexdigest()
            for split in results
        },
        "image_bytes": sum(row["bytes"] for items in results.values() for row in items),
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "warning": (
            "Selected member hashes are recorded; the complete source ZIP was not downloaded "
            "or SHA-512 checked locally."
        ),
    }
    (output / "fetch-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
