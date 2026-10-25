"""Acquire only page-matched Open Images pilot images to an external cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image, ImageOps

MAX_MEDIA_BYTES = 8 * 1024 * 1024
ALLOWED_HOST = re.compile(r"^(?:c[1-8]|farm[1-9]|live)\.staticflickr\.com$")
USER_AGENT = "Mozilla/5.0 (compatible; OpenImagesResearchPilot/1.0)"


class _MediaRedirectGuard(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlparse(urljoin(req.full_url, newurl))
        if target.scheme != "https" or not ALLOWED_HOST.fullmatch(target.hostname or ""):
            raise ValueError("media redirect left the approved HTTPS Flickr CDN hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(size: int = 32) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2 * positions[None, :] + 1) * frequencies / (2 * size))
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


_DCT32 = _dct_matrix()


def perceptual_hash(image: Image.Image) -> str:
    grayscale = (
        ImageOps.exif_transpose(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    )
    pixels = np.asarray(grayscale, dtype=np.float64)
    transformed = _DCT32 @ pixels @ _DCT32.T
    low_frequency = transformed[:8, :8].copy()
    median = float(np.median(low_frequency.reshape(-1)[1:]))
    bits = (low_frequency.reshape(-1) > median).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _decode_metadata(path: Path) -> tuple[str, str, int, int, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        width, height = normalized.size
        if width < 1 or height < 1:
            raise ValueError("decoded image has invalid dimensions")
        pixel_bytes = normalized.tobytes()
        normalized_hash = hashlib.sha256(
            f"RGB:{width}x{height}:".encode("ascii") + pixel_bytes
        ).hexdigest()
        return image.format or "unknown", normalized_hash, width, height, perceptual_hash(image)


def prepare_download_records(
    *,
    split: str,
    task_manifest: dict[str, Any],
    page_receipts: list[dict[str, Any]],
    head_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if split not in {"train", "validation"} or task_manifest.get("split") != split:
        raise ValueError("task manifest split mismatch")
    receipt_by_id = {row.get("ImageID"): row for row in page_receipts}
    head_by_id = {row.get("image_id"): row for row in head_records if row.get("split") == split}
    if len(receipt_by_id) != len(page_receipts) or len(head_by_id) != sum(
        row.get("split") == split for row in head_records
    ):
        raise ValueError("duplicate image IDs in rights or media HEAD receipts")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example in task_manifest.get("examples", []):
        image_id = example.get("image_id")
        if not isinstance(image_id, str) or image_id in seen:
            raise ValueError("task manifest image IDs must be unique")
        seen.add(image_id)
        rights = receipt_by_id.get(image_id)
        head = head_by_id.get(image_id)
        if rights is None or head is None:
            raise ValueError("task image lacks a rights or media availability receipt")
        if rights.get("landing_page_check_status") != "metadata_confirmed_on_landing_page":
            raise ValueError("task image did not pass the page-declaration gate")
        if head.get("status") != "ok" or head.get("content_type") != "image/jpeg":
            continue
        url = rights.get("Thumbnail300KURL") or rights.get("OriginalURL")
        parsed = urlparse(url or "")
        if parsed.scheme != "https" or not ALLOWED_HOST.fullmatch(parsed.hostname or ""):
            raise ValueError("image URL is outside the allowlisted HTTPS Flickr CDN")
        expected_photo_id = urlparse(rights["OriginalLandingURL"]).path.rstrip("/").split("/")[-1]
        media_photo_id = Path(parsed.path).name.split("_", 1)[0]
        if not expected_photo_id.isdecimal() or media_photo_id != expected_photo_id:
            raise ValueError("Flickr media URL does not identify the checked landing-page photo")
        records.append(
            {
                "split": split,
                "image_id": image_id,
                "sample_id": example["sample_id"],
                "target_label": example["target_label"],
                "target": example["target"],
                "options": example["options"],
                "target_position": example["target_position"],
                "url": url,
                "url_type": "thumbnail_300k" if rights.get("Thumbnail300KURL") else "original",
                "original_md5_base64": rights["OriginalMD5"],
                "landing_page_license": rights["landing_page_metadata"]["license_url"],
                "page_license_and_attribution_match": True,
                "individual_rights_verified": False,
            }
        )
    return records


def download_one(record: dict[str, Any], *, split_dir: Path) -> dict[str, Any]:
    path = split_dir / f"{record['image_id']}.jpg"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite media {path.name}")
    request = Request(record["url"], headers={"User-Agent": USER_AGENT})
    opener = build_opener(_MediaRedirectGuard)
    temporary: Path | None = None
    try:
        with opener.open(request, timeout=30) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or not ALLOWED_HOST.fullmatch(final.hostname or ""):
                raise ValueError("final media URL is outside the allowlisted CDN")
            content_type = response.headers.get_content_type()
            if content_type != "image/jpeg":
                raise ValueError(f"unexpected media Content-Type: {content_type}")
            declared_length = response.headers.get("Content-Length")
            if (
                declared_length
                and declared_length.isdigit()
                and int(declared_length) > MAX_MEDIA_BYTES
            ):
                raise ValueError("media exceeds the 8 MiB per-image cap")
            split_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{record['image_id']}.",
                suffix=".part",
                dir=split_dir,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                total = 0
                while chunk := response.read(64 * 1024):
                    total += len(chunk)
                    if total > MAX_MEDIA_BYTES:
                        raise ValueError("media stream exceeded the 8 MiB per-image cap")
                    stream.write(chunk)
            image_format, normalized_hash, width, height, p_hash = _decode_metadata(temporary)
            if image_format != "JPEG":
                raise ValueError(f"decoded image format is {image_format}, not JPEG")
            media_sha256 = _sha256(temporary)
            temporary.replace(path)
            temporary = None
            return record | {
                "status": "saved",
                "local_path": path.name,
                "bytes": total,
                "media_sha256": media_sha256,
                "normalized_rgb_sha256": normalized_hash,
                "perceptual_hash_64": p_hash,
                "width": width,
                "height": height,
                "decoded_format": image_format,
            }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return record | {
            "status": f"failed_{type(exc).__name__}",
            "error": str(exc)[:240],
            "local_path": None,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def audit_cross_split_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in rows if row.get("split") == "train" and row.get("status") == "saved"]
    validation = [
        row for row in rows if row.get("split") == "validation" and row.get("status") == "saved"
    ]
    train_ids = {row["image_id"] for row in train}
    validation_ids = {row["image_id"] for row in validation}
    train_hashes = {row["normalized_rgb_sha256"] for row in train}
    validation_hashes = {row["normalized_rgb_sha256"] for row in validation}
    train_phash = {row["perceptual_hash_64"] for row in train}
    validation_phash = {row["perceptual_hash_64"] for row in validation}
    near_matches = []
    for left in train:
        a = int(left["perceptual_hash_64"], 16)
        for right in validation:
            distance = (a ^ int(right["perceptual_hash_64"], 16)).bit_count()
            if distance <= 4:
                near_matches.append(
                    {
                        "train_image_id": left["image_id"],
                        "validation_image_id": right["image_id"],
                        "hamming_distance": distance,
                    }
                )
    return {
        "train_saved": len(train),
        "validation_saved": len(validation),
        "train_validation_image_id_overlap": len(train_ids & validation_ids),
        "train_validation_normalized_rgb_overlap": len(train_hashes & validation_hashes),
        "train_validation_exact_phash_overlap": len(train_phash & validation_phash),
        "train_validation_phash_hamming_at_most_4": near_matches,
        "media_bodies_saved": bool(train or validation),
        "individual_rights_verified": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retry-failed-from", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite media cache: {output}")
    root = args.artifact_root.resolve(strict=True)
    all_records = []
    prior_rows: list[dict[str, Any]] = []
    if args.retry_failed_from:
        prior_root = args.retry_failed_from.resolve(strict=True)
        for split in ("train", "validation"):
            previous = _read_jsonl(prior_root / f"{split}-media-manifest.jsonl")
            prior_rows.extend(previous)
        all_records = [row for row in prior_rows if row.get("status") != "saved"]
        if args.workers != 1:
            raise ValueError("use --workers 1 when retrying media requests")
    else:
        for split in ("train", "validation"):
            manifest = json.loads(
                (root / f"page-matched-{split}-tasks.json").read_text(encoding="utf-8")
            )
            receipts = _read_jsonl(root / f"private-{split}-page-checks.jsonl")
            head = [
                row
                for row in _read_jsonl(root / "private-media-head-preflight.jsonl")
                if row.get("split") == split
            ]
            all_records.extend(
                prepare_download_records(
                    split=split,
                    task_manifest=manifest,
                    page_receipts=receipts,
                    head_records=head,
                )
            )
    output.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, record, split_dir=output / record["split"]): record
            for record in all_records
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if args.retry_failed_from:
                time.sleep(2)
    results.sort(key=lambda row: (row["split"], row["image_id"]))
    for split in ("train", "validation"):
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in results if row["split"] == split
        )
        (output / f"{split}-media-manifest.jsonl").write_text(payload, encoding="utf-8")
    combined = [row for row in prior_rows if row.get("status") == "saved"] + [
        row for row in results if row.get("status") == "saved"
    ]
    summary = audit_cross_split_duplicates(combined) | {
        "input_rows": len(all_records),
        "retry_rows": len(all_records) if args.retry_failed_from else 0,
        "saved_this_attempt": sum(row["status"] == "saved" for row in results),
        "failed_downloads": sum(row["status"] != "saved" for row in results),
        "max_media_bytes": MAX_MEDIA_BYTES,
        "workers": args.workers,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
