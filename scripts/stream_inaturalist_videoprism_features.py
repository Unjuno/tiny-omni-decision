"""Extract pinned VideoPrism features while streaming CC0 iNaturalist media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.download_inaturalist_cc0_bird_media import (
        download_one,
        sha256_file,
        validate_splits,
    )
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from download_inaturalist_cc0_bird_media import download_one, sha256_file, validate_splits

CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
FEATURE_WIDTH = 768


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _old_media(paths: list[Path]) -> tuple[set[str], set[int], dict[str, str]]:
    exact: set[str] = set()
    perceptual: set[int] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        path = path.resolve(strict=True)
        hashes[str(path)] = sha256_file(path)
        rows = _read_rows(path)
        for row in rows:
            if row.get("status") not in (None, "saved"):
                raise ValueError(f"previous media manifest contains failed rows: {path}")
            for key in ("download_sha256", "media_sha256", "normalized_rgb_sha256"):
                value = row.get(key)
                if value:
                    exact.add(str(value))
            value = row.get("perceptual_hash_64")
            if value:
                perceptual.add(int(str(value), 16))
    if not exact or not perceptual:
        raise ValueError("previous media manifests need exact and perceptual hashes")
    return exact, perceptual, hashes


def _reject_media_overlap(
    row: dict[str, Any],
    old_exact: set[str],
    old_phashes: set[int],
    seen_exact: set[str],
    seen_phashes: list[tuple[int, str, str]],
) -> None:
    current_hashes = {
        str(row[key]) for key in ("download_sha256", "media_sha256", "normalized_rgb_sha256")
    }
    if current_hashes & old_exact or current_hashes & seen_exact:
        raise ValueError(f"exact media hash overlap at photo {row['photo_id']}")
    current_phash = int(str(row["perceptual_hash_64"]), 16)
    for previous, previous_split, previous_id in [
        *((value, "legacy", "legacy") for value in old_phashes),
        *seen_phashes,
    ]:
        if (current_phash ^ previous).bit_count() <= 4:
            raise ValueError(
                f"perceptual hash distance <=4 between {row['split']}:{row['photo_id']} "
                f"and {previous_split}:{previous_id}"
            )
    seen_exact.update(current_hashes)
    seen_phashes.append((current_phash, str(row["split"]), str(row["photo_id"])))


def _readiness_line(worker: subprocess.Popen[str]) -> dict[str, Any]:
    line = worker.stdout.readline() if worker.stdout is not None else ""
    if not line:
        raise RuntimeError("VideoPrism worker exited before its readiness receipt")
    result = json.loads(line)
    if (
        result.get("status") != "ready"
        or result.get("device") != ["TFRT_CPU_0"]
        or result.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or result.get("source_commit") != SOURCE_COMMIT
    ):
        raise RuntimeError(f"unexpected VideoPrism worker receipt: {result}")
    return result


def run(
    *,
    metadata_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    source_root: Path,
    worker_python: Path,
    previous_media_manifests: list[Path],
    minimum_free_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    metadata_dir = metadata_dir.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    worker_python = worker_python.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {source_commit}")
    free_bytes = shutil.disk_usage(output_dir.parent).free
    if free_bytes < minimum_free_bytes:
        raise OSError(f"free-space guard: {free_bytes} < {minimum_free_bytes} bytes")
    train = _read_rows(metadata_dir / "train.jsonl")
    validation = _read_rows(metadata_dir / "validation.jsonl")
    validate_splits(train, validation)
    class_ids = sorted({int(row["taxon_id"]) for row in train + validation})
    if len(class_ids) != 8 or {int(row["taxon_id"]) for row in train} != set(class_ids):
        raise ValueError("expected all eight taxa in train and validation")
    if {int(row["taxon_id"]) for row in validation} != set(class_ids):
        raise ValueError("validation is missing a class")
    old_exact, old_phashes, previous_hashes = _old_media(previous_media_manifests)
    metadata_hashes = {
        split: sha256_file(metadata_dir / f"{split}.jsonl") for split in ("train", "validation")
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    staging = output_dir / "_staging"
    staging.mkdir()
    arrays_path = output_dir / "features.npy"
    features = np.lib.format.open_memmap(
        arrays_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(train) + len(validation), FEATURE_WIDTH),
    )
    features[:] = np.nan
    worker_env = os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = ""
    worker_env["JAX_PLATFORMS"] = "cpu"
    stderr_path = output_dir / "videoprism-worker.stderr.log"
    started = time.perf_counter()
    attempts_started_utc = datetime.now(UTC).isoformat()
    run_config = {
        "metadata_directory": str(metadata_dir),
        "metadata_hashes": metadata_hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "video_model_revision": MODEL_REVISION,
        "video_source_commit": SOURCE_COMMIT,
        "source_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "previous_media_manifest_hashes": previous_hashes,
        "feature_width": FEATURE_WIDTH,
        "frames": 8,
        "frame_size": 288,
        "preprocessing": (
            "EXIF-oriented CC0 medium image, resize short side 288, center crop, "
            "repeat 8 frames, RGB float32 [0,1]"
        ),
        "pooling": "mean over all VideoPrism output tokens to 768 float32 values",
        "device_policy": "dedicated VideoPrism worker, JAX CPU only; no CUDA",
        "media_retention": (
            "stream each HTTPS response to a bounded temporary file; retain normalized "
            "PNG only until its feature is verified, then delete it; preserve hashes only"
        ),
        "minimum_free_bytes": minimum_free_bytes,
    }
    _write_json(output_dir / "run-config.json", run_config)
    _write_json(
        output_dir / "run-status.json",
        {"status": "RUNNING", "started_utc": attempts_started_utc, "completed_rows": 0},
    )
    media_out = (output_dir / "media-receipts.jsonl").open("w", encoding="utf-8", newline="\n")
    feature_out = (output_dir / "feature-rows.jsonl").open("w", encoding="utf-8", newline="\n")
    worker: subprocess.Popen[str] | None = None
    worker_receipt: dict[str, Any] = {}
    seen_exact: set[str] = set()
    seen_phashes: list[tuple[int, str, str]] = []
    elapsed_by_sample: list[float] = []
    all_rows = [("train", row) for row in train] + [("validation", row) for row in validation]
    completed_rows = 0
    try:
        with stderr_path.open("w", encoding="utf-8", newline="\n") as worker_stderr:
            worker = subprocess.Popen(
                [
                    str(worker_python),
                    "-u",
                    str(Path(__file__).with_name("inaturalist_videoprism_worker.py")),
                    "--source-root",
                    str(source_root),
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-sha256",
                    CHECKPOINT_SHA256,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=worker_stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=worker_env,
            )
            worker_receipt = _readiness_line(worker)
            for index, (split, row) in enumerate(all_rows):
                free_bytes = shutil.disk_usage(output_dir.parent).free
                if free_bytes < minimum_free_bytes:
                    raise OSError(
                        f"free-space guard before {split}:{row['photo_id']}: "
                        f"{free_bytes} < {minimum_free_bytes} bytes"
                    )
                split_dir = staging / split
                receipt = download_one(row | {"split": split}, split_dir)
                if receipt.get("status") != "saved":
                    raise RuntimeError(
                        f"media download failed for {split}:{row['photo_id']}: {receipt}"
                    )
                _reject_media_overlap(receipt, old_exact, old_phashes, seen_exact, seen_phashes)
                local_path = split_dir / str(receipt["local_path"])
                sample_id = f"inat:{int(row['taxon_id'])}:{row['observation_id']}"
                request = json.dumps({"sample_id": sample_id, "path": str(local_path)})
                if worker.stdin is None or worker.stdout is None:
                    raise RuntimeError("VideoPrism worker pipes closed unexpectedly")
                worker.stdin.write(request + "\n")
                worker.stdin.flush()
                response_line = worker.stdout.readline()
                if not response_line:
                    raise RuntimeError("VideoPrism worker exited during extraction")
                response = json.loads(response_line)
                vector = np.asarray(response.get("feature"), dtype=np.float32)
                if (
                    response.get("sample_id") != sample_id
                    or vector.shape != (FEATURE_WIDTH,)
                    or not np.isfinite(vector).all()
                ):
                    raise RuntimeError(f"invalid worker response for {sample_id}")
                features[index] = vector
                features.flush()
                elapsed = float(response["forward_seconds"])
                elapsed_by_sample.append(elapsed)
                feature_out.write(
                    json.dumps(
                        {
                            "index": index,
                            "sample_id": sample_id,
                            "split": split,
                            "taxon_id": int(row["taxon_id"]),
                            "label": str(row["label"]),
                            "observation_id": str(row["observation_id"]),
                            "photo_id": str(row["photo_id"]),
                            "observer_group_sha256": str(row["observer_group_sha256"]),
                            "feature_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                            "forward_seconds": elapsed,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                feature_out.flush()
                completed_rows += 1
                receipt["local_path"] = None
                receipt["media_retained"] = False
                media_out.write(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
                media_out.flush()
                local_path.unlink()
                if index % 25 == 24 or index == len(all_rows) - 1:
                    features.flush()
                    progress = {
                        "status": "RUNNING",
                        "completed_rows": index + 1,
                        "total_rows": len(all_rows),
                        "elapsed_seconds": time.perf_counter() - started,
                        "last_sample_id": sample_id,
                    }
                    _write_json(output_dir / "run-status.json", progress)
                    print(json.dumps(progress), flush=True)
            worker.stdin.write('{"command":"stop"}\n')
            worker.stdin.flush()
            worker.stdin.close()
            return_code = worker.wait(timeout=30)
            if return_code:
                raise RuntimeError(f"VideoPrism worker exited with status {return_code}")
    except Exception as error:
        _write_json(
            output_dir / "run-status.json",
            {
                "status": "FAILED",
                "started_utc": attempts_started_utc,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_rows": completed_rows,
                "error": f"{type(error).__name__}: {error}",
                "worker_stderr": str(stderr_path),
            },
        )
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.wait(timeout=30)
        raise
    finally:
        media_out.close()
        feature_out.close()
        del features

    feature_rows = _read_rows(output_dir / "feature-rows.jsonl")
    media_rows = _read_rows(output_dir / "media-receipts.jsonl")
    if len(feature_rows) != len(all_rows) or len(media_rows) != len(all_rows):
        raise RuntimeError("final row count does not match the fixed metadata cohort")
    if not np.isfinite(np.load(arrays_path, mmap_mode="r")).all():
        raise RuntimeError("feature cache contains non-finite values")
    classes = {
        str(class_id): {
            split: sum(
                row["split"] == split and int(row["taxon_id"]) == class_id for row in feature_rows
            )
            for split in ("train", "validation")
        }
        for class_id in sorted({int(row["taxon_id"]) for row in feature_rows})
    }
    result = {
        "status": "complete",
        "started_utc": attempts_started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "total_wall_seconds": time.perf_counter() - started,
        "worker": worker_receipt,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "unique_media_assets": len(media_rows),
        "train_validation_exact_media_overlap": 0,
        "train_validation_phash_hamming_at_most_4": 0,
        "legacy_exact_hash_overlap": 0,
        "legacy_phash_hamming_at_most_4": 0,
        "per_class": classes,
        "mean_forward_seconds": float(np.mean(elapsed_by_sample)),
        "feature_cache_file": arrays_path.name,
        "feature_cache_sha256": sha256_file(arrays_path),
        "feature_rows_sha256": sha256_file(output_dir / "feature-rows.jsonl"),
        "media_receipts_sha256": sha256_file(output_dir / "media-receipts.jsonl"),
        "metadata_sha256": metadata_hashes,
        "config_sha256": hashlib.sha256(
            json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__).resolve()),
            "inaturalist_videoprism_worker.py": sha256_file(
                Path(__file__).with_name("inaturalist_videoprism_worker.py")
            ),
        },
        "class_counts": classes,
    }
    _write_json(output_dir / "result.json", result)
    _write_json(
        output_dir / "run-status.json",
        {
            "status": "COMPLETE",
            "completed_rows": len(all_rows),
            "result_sha256": sha256_file(output_dir / "result.json"),
        },
    )
    (staging / "train").rmdir()
    (staging / "validation").rmdir()
    staging.rmdir()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--previous-media-manifest", type=Path, action="append", required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    result = run(
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        source_root=args.source_root,
        worker_python=args.worker_python,
        previous_media_manifests=args.previous_media_manifest,
        minimum_free_bytes=args.minimum_free_bytes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
