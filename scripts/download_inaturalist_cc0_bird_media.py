"""Download only the frozen, per-record CC0 iNaturalist bird cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

ALLOWED_HOST = "inaturalist-open-data.s3.amazonaws.com"
USER_AGENT = "tiny-omni-decision-research-preflight/1.0"
MAX_MEDIA_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
PHOTO_PATH = re.compile(r"^/photos/(\d+)/square\.(?P<extension>jpg|jpeg|png)$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def medium_url(row: dict[str, Any]) -> str:
    parsed = urlparse(str(row.get("photo_url", "")))
    match = PHOTO_PATH.fullmatch(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_HOST
        or parsed.port not in (None, 443)
        or not match
        or match.group(1) != str(row.get("photo_id"))
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("photo URL is not the expected CC0 iNaturalist square asset")
    return parsed._replace(path=parsed.path.replace("/square.", "/medium.")).geturl()


class _RedirectGuard(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(urljoin(req.full_url, newurl))
        if (
            parsed.scheme != "https"
            or parsed.hostname != ALLOWED_HOST
            or parsed.port not in (None, 443)
        ):
            raise ValueError("media redirect left the approved HTTPS iNaturalist CDN")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_splits(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    for split, rows in (("train", train), ("validation", validation)):
        if not rows:
            raise ValueError(f"{split} manifest is empty")
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"row assigned to the wrong {split} manifest")
            if row.get("observation_license") != "cc0" or row.get("photo_license") != "cc0":
                raise ValueError("both observation and selected photo must explicitly be CC0")
            if not row.get("observer_group_sha256") or not row.get("observation_id"):
                raise ValueError("source and observer-group identities are required")
            if not row.get("label") or not isinstance(row.get("taxon_id"), int):
                raise ValueError("every row needs a class label and integer taxon ID")
            medium_url(row)
        for key in ("observation_id", "photo_id"):
            values = [str(row[key]) for row in rows]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {key} in {split}")
    for key in ("observation_id", "photo_id", "observer_group_sha256"):
        if {str(row[key]) for row in train} & {str(row[key]) for row in validation}:
            raise ValueError(f"train/validation {key} overlap")


def _perceptual_hash(image: Any) -> str:
    from PIL import Image, ImageOps
    positions = np.arange(32, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2 * positions[None, :] + 1) * frequencies / 64)
    matrix[0] *= math.sqrt(1 / 32)
    matrix[1:] *= math.sqrt(2 / 32)
    grayscale = ImageOps.exif_transpose(image).convert("L").resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    transformed = matrix @ np.asarray(grayscale, dtype=np.float64) @ matrix.T
    low_frequency = transformed[:8, :8].reshape(-1)
    median = float(np.median(low_frequency[1:]))
    value = 0
    for bit in low_frequency > median:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _decode_and_strip(path: Path) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image_format = image.format
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        width, height = normalized.size
        if width < 32 or height < 32 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("decoded image dimensions are outside the allowed range")
        pixels = normalized.tobytes()
        normalized_hash = sha256_bytes(f"RGB:{width}x{height}:".encode() + pixels)
        return normalized, {
            "normalized_rgb_sha256": normalized_hash,
            "perceptual_hash_64": _perceptual_hash(normalized),
            "width": width,
            "height": height,
            "decoded_format": image_format,
        }


def download_one(row: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    url = medium_url(row)
    output = split_dir / f"{row['photo_id']}.png"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing photo {output.name}")
    opener = build_opener(_RedirectGuard)
    expected_media_type = "image/png" if url.endswith(".png") else "image/jpeg"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": expected_media_type})
    temporary: Path | None = None
    try:
        split_dir.mkdir(parents=True, exist_ok=True)
        with opener.open(request, timeout=45) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != ALLOWED_HOST:
                raise ValueError("final media URL left the approved HTTPS iNaturalist CDN")
            if response.headers.get_content_type() != expected_media_type:
                raise ValueError("media response does not match the source image format")
            declared_length = response.headers.get("Content-Length")
            if (
                declared_length
                and declared_length.isdigit()
                and int(declared_length) > MAX_MEDIA_BYTES
            ):
                raise ValueError("media exceeds the 8 MiB per-photo limit")
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".inat-", suffix=".part", dir=split_dir, delete=False
            ) as stream:
                temporary = Path(stream.name)
                size = 0
                while block := response.read(64 * 1024):
                    size += len(block)
                    if size > MAX_MEDIA_BYTES:
                        raise ValueError("media stream exceeded the 8 MiB per-photo limit")
                    stream.write(block)
        raw_hash = sha256_file(temporary)
        normalized, metadata = _decode_and_strip(temporary)
        output.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(output, format="PNG", optimize=False)
        media_hash = sha256_file(output)
        result = row | {
            "status": "saved",
            "local_path": output.name,
            "medium_url": url,
            "downloaded_bytes": size,
            "download_sha256": raw_hash,
            "media_sha256": media_hash,
            **metadata,
            "exif_stripped": True,
        }
        return result
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        output.unlink(missing_ok=True)
        return row | {
            "status": f"failed_{type(error).__name__}",
            "error": str(error)[:200],
            "local_path": None,
            "medium_url": url,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _cross_split_media_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    saved = [row for row in rows if row.get("status") == "saved"]
    train = [row for row in saved if row["split"] == "train"]
    validation = [row for row in saved if row["split"] == "validation"]
    exact = {
        key: len({row[key] for row in train} & {row[key] for row in validation})
        for key in ("download_sha256", "media_sha256", "normalized_rgb_sha256")
    }
    near = []
    for left in train:
        a = int(left["perceptual_hash_64"], 16)
        for right in validation:
            distance = (a ^ int(right["perceptual_hash_64"], 16)).bit_count()
            if distance <= 4:
                near.append(
                    {
                        "train_photo_id": left["photo_id"],
                        "validation_photo_id": right["photo_id"],
                        "hamming_distance": distance,
                    }
                )
    return {
        "train_saved": len(train),
        "validation_saved": len(validation),
        "cross_split_exact_hash_overlaps": exact,
        "cross_split_phash_hamming_at_most_4": near,
        "media_identity_gate_pass": not any(exact.values()),
    }


def run(metadata_dir: Path, output_dir: Path, workers: int = 2) -> dict[str, Any]:
    if workers not in (1, 2):
        raise ValueError("workers must be one or two")
    metadata_dir = metadata_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite media cache: {output_dir}")
    train = _read_jsonl(metadata_dir / "train.jsonl")
    validation = _read_jsonl(metadata_dir / "validation.jsonl")
    validate_splits(train, validation)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        {**row, "split": split}
        for split, values in (("train", train), ("validation", validation))
        for row in values
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, row, output_dir / row["split"]): row for row in rows
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"processed media {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: (row["split"], int(row["taxon_id"]), row["observation_id"]))
    for split in ("train", "validation"):
        payload = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in results
            if row["split"] == split
        )
        (output_dir / f"{split}-media-manifest.jsonl").write_text(payload, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "source_metadata_dir": str(metadata_dir),
        "metadata_manifest_sha256": {
            split: sha256_file(metadata_dir / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "input_counts": {
            split: len(values)
            for split, values in (("train", train), ("validation", validation))
        },
        "downloaded": sum(row.get("status") == "saved" for row in results),
        "failed": sum(row.get("status") != "saved" for row in results),
        "worker_count": workers,
        "max_media_bytes": MAX_MEDIA_BYTES,
        "media_size_variant": "iNaturalist medium (max 500px), derived from the API photo URL",
        "audit": _cross_split_media_audit(results),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.metadata_dir, args.output_dir, args.workers), indent=2))


if __name__ == "__main__":
    main()




