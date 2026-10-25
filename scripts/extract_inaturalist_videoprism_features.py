"""Extract pinned frozen VideoPrism-B features for an iNaturalist CC0 image cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
ENCODER_PARAMETERS = 114_365_184
FEATURE_WIDTH = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_split(
    metadata_dir: Path,
    media_dir: Path,
    split: str,
) -> list[dict[str, Any]]:
    source_rows = _read_jsonl(metadata_dir / f"{split}.jsonl")
    media_rows = _read_jsonl(media_dir / f"{split}-media-manifest.jsonl")
    if not source_rows or len(source_rows) != len(media_rows):
        raise ValueError(f"{split} source and media manifests must have identical nonzero rows")
    by_observation = {str(row.get("observation_id")): row for row in media_rows}
    if len(by_observation) != len(media_rows):
        raise ValueError(f"duplicate observation ID in {split} media manifest")
    prepared = []
    for source in source_rows:
        observation_id = str(source["observation_id"])
        media = by_observation.get(observation_id)
        if media is None or media.get("status") != "saved":
            raise ValueError(f"missing successful media for one {split} source row")
        for key in (
            "photo_id",
            "taxon_id",
            "label",
            "observation_license",
            "photo_license",
            "observer_group_sha256",
        ):
            if media.get(key) != source.get(key):
                raise ValueError(f"source/media mismatch for {split} field {key}")
        if source.get("split") != split or media.get("split") != split:
            raise ValueError(f"source/media split mismatch for {split}")
        if source.get("observation_license") != "cc0" or source.get("photo_license") != "cc0":
            raise ValueError("feature extraction requires explicit observation and photo CC0")
        relative_path = Path(str(media.get("local_path", "")))
        media_path = (media_dir / split / relative_path).resolve(strict=True)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or media_path.parent != (media_dir / split).resolve(strict=True)
        ):
            raise ValueError("media path must remain a file directly inside its frozen split")
        if sha256_file(media_path) != media.get("media_sha256"):
            raise ValueError(f"media bytes do not match frozen SHA-256 for split {split}")
        if media.get("exif_stripped") is not True:
            raise ValueError("media must be EXIF-stripped before feature extraction")
        prepared.append(
            {
                **source,
                "sample_id": f"inat:{source['taxon_id']}:{observation_id}",
                "media_path": str(media_path),
                "download_sha256": media["download_sha256"],
                "media_sha256": media["media_sha256"],
                "normalized_rgb_sha256": media["normalized_rgb_sha256"],
                "perceptual_hash_64": media["perceptual_hash_64"],
                "width": media["width"],
                "height": media["height"],
            }
        )
    prepared.sort(key=lambda row: (int(row["taxon_id"]), str(row["observation_id"])))
    if len({row["sample_id"] for row in prepared}) != len(prepared):
        raise ValueError(f"duplicate sample ID in {split}")
    return prepared


def validate_feature_splits(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    for key in (
        "observation_id",
        "photo_id",
        "observer_group_sha256",
        "download_sha256",
        "media_sha256",
        "normalized_rgb_sha256",
    ):
        if {str(row[key]) for row in train} & {str(row[key]) for row in validation}:
            raise ValueError(f"train/validation {key} overlap")
    near = []
    for left in train:
        left_hash = int(left["perceptual_hash_64"], 16)
        for right in validation:
            distance = (left_hash ^ int(right["perceptual_hash_64"], 16)).bit_count()
            if distance <= 4:
                near.append(distance)
                if len(near) == 5:
                    break
        if len(near) == 5:
            break
    if near:
        raise ValueError("near-duplicate train/validation images at pHash Hamming distance <=4")


def _extract_rows(
    rows: list[dict[str, Any]],
    process: subprocess.Popen[str],
    split: str,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("VideoPrism worker pipes were not opened")
    features = []
    records = []
    for index, row in enumerate(rows, start=1):
        request_id = f"{split}:{index}"
        request = {
            "request_id": request_id,
            "task": "encode_image",
            "payload": {
                "image_path": row["media_path"],
                "expected_media_sha256": row["media_sha256"],
            },
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"VideoPrism worker exited during {request_id}")
        response = json.loads(line)
        if response.get("request_id") != request_id:
            raise ValueError(f"worker response order mismatch at {request_id}")
        if "error" in response:
            raise RuntimeError(f"worker failed during {request_id}: {response['error']}")
        feature = np.asarray(response.get("image_feature"), dtype=np.float32)
        if feature.shape != (FEATURE_WIDTH,) or not np.isfinite(feature).all():
            raise ValueError(f"invalid VideoPrism feature during {request_id}")
        record = row | {
            "feature_sha256": hashlib.sha256(feature.tobytes()).hexdigest(),
            "encoder_metrics": response.get("encoder_metrics", {}),
        }
        features.append(feature)
        records.append(record)
        if index == 1 or index % 100 == 0 or index == len(rows):
            np.savez_compressed(
                output_dir / f"{split}-features.partial.npz",
                sample_ids=np.asarray([item["sample_id"] for item in records]),
                features=np.stack(features).astype(np.float32, copy=False),
                completed_rows=np.asarray([index], dtype=np.int32),
            )
            print(f"encoded {split} {index}/{len(rows)}", flush=True)
    return np.stack(features), records


def run(
    metadata_dir: Path,
    media_dir: Path,
    assembly: Path,
    source_root: Path,
    python: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started_counter = time.perf_counter()
    metadata_dir, media_dir = metadata_dir.resolve(strict=True), media_dir.resolve(strict=True)
    assembly, source_root, python = (
        assembly.resolve(strict=True),
        source_root.resolve(strict=True),
        python.resolve(strict=True),
    )
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(ROOT):
        raise ValueError("feature artifacts and source IDs must stay outside the Git repository")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output_dir}")
    image_component = assembly / "components/image"
    checkpoint = image_component / "videoprism-base-f16r288.npz"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {upstream_commit}")
    train = prepare_split(metadata_dir, media_dir, "train")
    validation = prepare_split(metadata_dir, media_dir, "validation")
    validate_feature_splits(train, validation)

    classes = sorted(
        {int(row["taxon_id"]): str(row["label"]) for row in train + validation}.items()
    )
    class_ids = [taxon_id for taxon_id, _ in classes]
    class_names = [label for _, label in classes]
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "experiment_id": "inat-birds-videoprism-frozen-features-v1",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "extractor_sha256": sha256_file(Path(__file__)),
        "worker_sha256": sha256_file(ROOT / "scripts/path_a_videoprism_worker.py"),
        "input_hashes": {
            "metadata_train": sha256_file(metadata_dir / "train.jsonl"),
            "metadata_validation": sha256_file(metadata_dir / "validation.jsonl"),
            "media_train": sha256_file(media_dir / "train-media-manifest.jsonl"),
            "media_validation": sha256_file(media_dir / "validation-media-manifest.jsonl"),
            "media_summary": sha256_file(media_dir / "summary.json"),
        },
        "video_model_revision": MODEL_REVISION,
        "video_source_commit": upstream_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "encoder_parameters": ENCODER_PARAMETERS,
        "feature_shape": [FEATURE_WIDTH],
        "input_preprocessing": (
            "EXIF-oriented RGB; resize short side 288, center crop 288, repeat over 8 frames, "
            "values in [0,1]"
        ),
        "pooling": "mean across all output tokens to 768 dimensions",
        "device_policy": "JAX CPU only, CUDA_VISIBLE_DEVICES empty and JAX_PLATFORMS=cpu",
        "split_counts": {"train": len(train), "validation": len(validation)},
        "class_ids": class_ids,
        "class_names": class_names,
        "sealed_audit_loaded": False,
        "model_training_performed": False,
    }
    _write_json(output_dir / "config.json", metadata)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "PYTHONUTF8": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "OMP_NUM_THREADS": "4",
            "TF_NUM_INTRAOP_THREADS": "4",
            "TF_NUM_INTEROP_THREADS": "2",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    command = [
        str(python),
        str(ROOT / "scripts/path_a_videoprism_worker.py"),
        "--assembly",
        str(assembly),
        "--source-root",
        str(source_root),
    ]
    worker = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
    )
    try:
        if worker.stdout is None:
            raise RuntimeError("VideoPrism worker stdout was not opened")
        ready_line = worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"VideoPrism worker exited during startup ({worker.poll()})")
        ready = json.loads(ready_line)
        if (
            not ready.get("ready")
            or ready.get("profile") != "videoprism_image_video_encoder"
            or ready.get("video_model_revision") != MODEL_REVISION
            or ready.get("video_checkpoint_sha256") != CHECKPOINT_SHA256
            or ready.get("video_encoder_parameters") != ENCODER_PARAMETERS
            or ready.get("devices") != ["TFRT_CPU_0"]
        ):
            raise ValueError(f"unexpected CPU VideoPrism worker handshake: {ready}")
        metadata["worker"] = ready
        _write_json(output_dir / "config.json", metadata)
        total_forward_seconds = 0.0
        feature_hashes = {}
        sample_order_hashes = {}
        for split, rows in (("train", train), ("validation", validation)):
            features, records = _extract_rows(rows, worker, split, output_dir)
            targets = np.asarray(
                [class_ids.index(int(row["taxon_id"])) for row in rows], dtype=np.int32
            )
            feature_path = output_dir / f"{split}-features.npz"
            np.savez_compressed(
                feature_path,
                sample_ids=np.asarray([row["sample_id"] for row in rows]),
                observation_ids=np.asarray([row["observation_id"] for row in rows]),
                photo_ids=np.asarray([row["photo_id"] for row in rows]),
                features=features.astype(np.float32, copy=False),
                targets=targets,
                class_ids=np.asarray(class_ids, dtype=np.int64),
                feature_sha256=np.asarray([row["feature_sha256"] for row in records]),
            )
            row_path = output_dir / f"{split}-rows.jsonl"
            row_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                    for row in records
                ),
                encoding="utf-8",
            )
            (output_dir / f"{split}-features.partial.npz").unlink()
            feature_hashes[split] = sha256_file(feature_path)
            sample_order_hashes[split] = hashlib.sha256(
                "\n".join(row["sample_id"] for row in rows).encode()
            ).hexdigest()
            total_forward_seconds += sum(
                float(row["encoder_metrics"].get("forward_seconds", 0.0)) for row in records
            )
        if worker.stdin is None:
            raise RuntimeError("VideoPrism worker stdin was not opened")
        worker.stdin.write('{"command":"shutdown"}\n')
        worker.stdin.flush()
        stopped_line = worker.stdout.readline()
        if not stopped_line or json.loads(stopped_line).get("stopped") is not True:
            raise RuntimeError("VideoPrism worker did not acknowledge shutdown")
        worker.wait(timeout=30)
        if worker.returncode != 0:
            raise RuntimeError(f"VideoPrism worker exited with status {worker.returncode}")
        metadata.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(UTC).isoformat(),
                "total_wall_seconds": time.perf_counter() - started_counter,
                "encoder_forward_seconds_sum": total_forward_seconds,
                "feature_cache_sha256": feature_hashes,
                "sample_id_order_sha256": sample_order_hashes,
                "output_hashes": {
                    path.name: sha256_file(path)
                    for path in output_dir.iterdir()
                    if path.is_file() and path.name != "config.json"
                },
            }
        )
    except Exception as error:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=30)
        metadata.update(
            {
                "status": "failed",
                "failed_utc": datetime.now(UTC).isoformat(),
                "failure": f"{type(error).__name__}: {error}",
                "total_wall_seconds": time.perf_counter() - started_counter,
            }
        )
        _write_json(output_dir / "config.json", metadata)
        raise
    _write_json(output_dir / "config.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.metadata_dir,
                args.media_dir,
                args.assembly,
                args.source_root,
                args.python,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
