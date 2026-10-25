"""Extract pinned frozen VideoPrism features for the page-matched OI pilot."""

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
EXPECTED_CHECKPOINT = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
EXPECTED_ENCODER_PARAMS = 114_365_184
FEATURE_SIZE = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_tasks(path: Path, expected_split: str) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") != expected_split:
        raise ValueError(f"wrong split in task manifest: {path}")
    rows = manifest.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty task manifest: {path}")
    by_id = {str(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate image IDs in {path}")
    return by_id


def _media_by_id(roots: list[Path], split: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for root in roots:
        for row in _jsonl(root / f"{split}-media-manifest.jsonl"):
            if row.get("status") != "saved":
                continue
            image_id = str(row["image_id"])
            current = by_id.get(image_id)
            if current is not None:
                if current.get("media_sha256") != row.get("media_sha256"):
                    raise ValueError(f"conflicting downloaded bytes for image {image_id}")
                continue
            row = row | {"media_root": str(root.resolve(strict=True))}
            by_id[image_id] = row
    return by_id


def prepare_split(
    task_path: Path,
    media_roots: list[Path],
    split: str,
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    tasks = _read_tasks(task_path, split)
    media = _media_by_id(media_roots, split)
    rows = []
    for image_id, task in tasks.items():
        item = media.get(image_id)
        if item is None:
            continue
        if item.get("page_license_and_attribution_match") is not True:
            raise ValueError(f"image {image_id} lacks the page-declaration match")
        if item.get("individual_rights_verified") is not False:
            raise ValueError("rights verification state must remain explicitly unverified")
        for key in ("sample_id", "target", "target_label", "options"):
            if item.get(key) != task.get(key):
                raise ValueError(f"task/media mismatch for {image_id}: {key}")
        media_path = Path(item["media_root"]) / split / str(item["local_path"])
        if not media_path.is_file() or sha256_file(media_path) != item.get("media_sha256"):
            raise ValueError(f"missing or modified cached image: {image_id}")
        rows.append(
            {
                **task,
                "media_path": str(media_path.resolve(strict=True)),
                "media_sha256": item["media_sha256"],
                "normalized_rgb_sha256": item["normalized_rgb_sha256"],
                "original_md5_base64": item["original_md5_base64"],
                "perceptual_hash_64": item["perceptual_hash_64"],
                "page_license_and_attribution_match": True,
                "individual_rights_verified": False,
            }
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no downloaded task rows available for {split}")
    return rows


def validate_splits(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    for name, rows in (("train", train), ("validation", validation)):
        if len({row["image_id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate image identity in {name}")
    overlap_fields = (
        "image_id",
        "original_md5_base64",
        "normalized_rgb_sha256",
    )
    for field in overlap_fields:
        left = {row[field] for row in train}
        right = {row[field] for row in validation}
        if left & right:
            raise ValueError(f"train/validation {field} overlap")
    nearby = []
    for left in train:
        a = int(left["perceptual_hash_64"], 16)
        for right in validation:
            distance = (a ^ int(right["perceptual_hash_64"], 16)).bit_count()
            if distance <= 4:
                nearby.append((left["image_id"], right["image_id"], distance))
    if nearby:
        raise ValueError(f"near-duplicate train/validation images: {nearby[:5]}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _extract(rows: list[dict[str, Any]], process: subprocess.Popen[str], split: str) -> np.ndarray:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("worker streams were not opened")
    values = []
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
            raise RuntimeError(f"VideoPrism worker error at {request_id}: {response['error']}")
        feature = np.asarray(response.get("image_feature"), dtype=np.float32)
        if feature.shape != (FEATURE_SIZE,) or not np.isfinite(feature).all():
            raise ValueError(f"invalid feature returned for {row['image_id']}")
        row["feature_sha256"] = hashlib.sha256(feature.tobytes()).hexdigest()
        row["encoder_metrics"] = response.get("encoder_metrics", {})
        values.append(feature)
        if index == 1 or index % 16 == 0 or index == len(rows):
            print(f"encoded {split} {index}/{len(rows)}", flush=True)
    return np.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--media-attempt-1", type=Path, required=True)
    parser.add_argument("--media-attempt-2", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()

    task_root = args.task_root.resolve(strict=True)
    media_roots = [
        args.media_attempt_1.resolve(strict=True),
        args.media_attempt_2.resolve(strict=True),
    ]
    assembly = args.assembly.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    python = args.python.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("features and source-identified rows must stay outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output}")
    checkpoint = assembly / "components/image/videoprism-base-f16r288.npz"
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("VideoPrism checkpoint does not match pinned revision")
    if not (assembly / "components/image/bundle-manifest.json").is_file():
        raise FileNotFoundError("pinned Path A image component bundle is incomplete")

    split_rows = {}
    for split in ("train", "validation"):
        split_rows[split] = prepare_split(
            task_root / f"page-matched-{split}-tasks.json",
            media_roots,
            split,
            limit=args.limit_per_split,
        )
    validate_splits(split_rows["train"], split_rows["validation"])

    output.mkdir(parents=True)
    input_paths = [
        task_root / "page-matched-train-tasks.json",
        task_root / "page-matched-validation-tasks.json",
        *(
            root / f"{split}-media-manifest.jsonl"
            for root in media_roots
            for split in ("train", "validation")
        ),
    ]
    config = {
        "schema_version": 1,
        "purpose": (
            "frozen VideoPrism image features for page-matched Open Images V6 "
            "hierarchy-choice pilot"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "extractor_sha256": sha256_file(Path(__file__)),
        "worker_sha256": sha256_file(ROOT / "scripts/path_a_videoprism_worker.py"),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "encoder_parameters": EXPECTED_ENCODER_PARAMS,
        "feature_shape_per_image": [FEATURE_SIZE],
        "device_policy": "JAX CPU only; CUDA_VISIBLE_DEVICES empty and JAX_PLATFORMS=cpu",
        "limits_per_split": args.limit_per_split,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_ids_sha256": {
            split: hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode()).hexdigest()
            for split, rows in split_rows.items()
        },
        "cross_split_overlap": {
            "image_id": 0,
            "original_md5_base64": 0,
            "normalized_rgb_sha256": 0,
            "perceptual_hash_hamming_at_most_4": 0,
        },
        "rights_note": (
            "Flickr page declaration and attribution profile matched; individual "
            "rights/ownership were not independently verified."
        ),
        "sealed_audit_loaded": False,
        "model_training_performed": False,
    }
    _write_json(output / "config.json", config)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "OMP_NUM_THREADS": "4",
            "TF_NUM_INTRAOP_THREADS": "4",
            "TF_NUM_INTEROP_THREADS": "2",
        }
    )
    project_paths = (str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(part for part in project_paths if part)
    command = [
        str(python),
        str(ROOT / "scripts/path_a_videoprism_worker.py"),
        "--assembly",
        str(assembly),
        "--source-root",
        str(source_root),
    ]
    print("starting pinned VideoPrism worker on CPU", flush=True)
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
    started = time.perf_counter()
    try:
        if worker.stdout is None:
            raise RuntimeError("worker stdout was not opened")
        ready_line = worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"VideoPrism worker exited during startup ({worker.poll()})")
        ready = json.loads(ready_line)
        if not ready.get("ready") or ready.get("profile") != "videoprism_image_video_encoder":
            raise ValueError("unexpected VideoPrism worker handshake")
        if ready.get("video_checkpoint_sha256") != EXPECTED_CHECKPOINT:
            raise ValueError("worker loaded an unexpected checkpoint")
        if ready.get("video_model_revision") != MODEL_REVISION:
            raise ValueError("worker loaded an unexpected model revision")
        if ready.get("video_encoder_parameters") != EXPECTED_ENCODER_PARAMS:
            raise ValueError("worker reports an unexpected parameter count")
        devices = ready.get("devices", [])
        if not devices or any("CPU" not in str(device).upper() for device in devices):
            raise ValueError(f"worker did not select CPU-only devices: {devices}")
        if any(
            accelerator in str(device).upper()
            for device in devices
            for accelerator in ("GPU", "CUDA", "TPU")
        ):
            raise ValueError(f"worker exposed an accelerator device: {devices}")
        config["worker"] = ready
        _write_json(output / "config.json", config)
        features = {}
        records = {}
        for split in ("train", "validation"):
            rows = split_rows[split]
            features[split] = _extract(rows, worker, split)
            records[split] = rows
            cache_path = output / f"{split}-features.npz"
            np.savez_compressed(
                cache_path,
                sample_ids=np.asarray([row["sample_id"] for row in rows]),
                image_ids=np.asarray([row["image_id"] for row in rows]),
                features=features[split],
                targets=np.asarray([row["target"] for row in rows]),
                target_labels=np.asarray([row["target_label"] for row in rows]),
                options_json=np.asarray([json.dumps(row["options"]) for row in rows]),
                media_sha256=np.asarray([row["media_sha256"] for row in rows]),
                feature_sha256=np.asarray([row["feature_sha256"] for row in rows]),
            )
            config.setdefault("cache_sha256", {})[split] = sha256_file(cache_path)
            _write_json(output / "config.json", config)
            (output / f"{split}-rows.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
        config["elapsed_seconds"] = time.perf_counter() - started
        config["feature_counts"] = {key: len(value) for key, value in features.items()}
        config["output_hashes"] = {
            path.name: sha256_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "config.json"
        }
        _write_json(output / "config.json", config)
        process = worker
        if process.stdin is None:
            raise RuntimeError("worker stdin was not opened")
        process.stdin.write('{"command":"shutdown"}\n')
        process.stdin.flush()
        stopped = process.stdout.readline()
        if json.loads(stopped).get("stopped") is not True:
            raise RuntimeError("worker did not acknowledge shutdown")
        process.wait(timeout=30)
        if process.returncode != 0:
            raise RuntimeError(f"VideoPrism worker exited with {process.returncode}")
    except BaseException:
        if worker.poll() is None and worker.stdin is not None:
            try:
                worker.stdin.write('{"command":"shutdown"}\n')
                worker.stdin.flush()
                worker.wait(timeout=20)
            except Exception:
                worker.terminate()
                worker.wait(timeout=10)
        raise
    print(json.dumps({"output": str(output), "config": config}, indent=2), flush=True)


if __name__ == "__main__":
    main()
