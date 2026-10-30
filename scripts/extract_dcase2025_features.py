"""Extract frozen synchronized audio/video features for DCASE 2025 dev-train."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.preflight_dcase2025 import preflight, sha256_file
from tiny_omni_decision.dcase2025 import load_dcase2025_devtrain_queries

VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
AUDIO_CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
AUDIO_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
BATCH_SIZE = 8


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def send_request(process: subprocess.Popen[str], request: dict[str, object]) -> dict[str, object]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("worker pipes are not available")
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    if not response:
        raise RuntimeError(f"worker exited before replying (exit={process.poll()})")
    value = json.loads(response)
    if value.get("status") != "PASS":
        raise RuntimeError(f"worker failure: {value}")
    return value


def start_worker(command: list[str], env: dict[str, str], log_path: Path):
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
    )
    response = process.stdout.readline() if process.stdout is not None else ""
    if not response:
        process.kill()
        log.close()
        raise RuntimeError(f"worker did not start; see {log_path}")
    ready = json.loads(response)
    if ready.get("status") != "READY":
        process.kill()
        log.close()
        raise RuntimeError(f"worker readiness check failed: {ready}")
    return process, log, ready


def query_manifest_rows(queries):
    return [
        {
            "sample_id": row.sample_id,
            "clip_id": row.clip_id,
            "parent_recording": row.parent_recording,
            "room_id": row.room_id,
            "view_degrees": row.view_degrees,
            "start_frame": row.start_frame,
            "event_frame": row.event_frame,
            "event_class": row.event_class,
            "target_onscreen": row.target_onscreen,
        }
        for row in queries
    ]


def run(args) -> dict[str, object]:
    output = args.output_dir.resolve()
    if output.exists() and not (output / "progress.json").is_file():
        raise FileExistsError("output path exists without resumable DCASE progress metadata")
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / "scratch"
    scratch.mkdir(exist_ok=True)

    expected_preflight = preflight(args.metadata_zip, args.media_manifest, seed=17)
    if expected_preflight["status"] != "PASS":
        raise ValueError("DCASE media/metadata preflight did not pass")
    with zipfile_for(args.metadata_zip) as archive:
        queries, inventory = load_dcase2025_devtrain_queries(archive, seed=17)
    query_rows = query_manifest_rows(queries)
    query_ids_hash = hashlib.sha256(
        "".join(f"{row['sample_id']}\n" for row in query_rows).encode()
    ).hexdigest()
    if query_ids_hash != expected_preflight["selected_sample_id_order_sha256"]:
        raise ValueError("query order differs from the frozen preflight")
    total = len(query_rows)

    run_identity = {
        "query_ids_sha256": query_ids_hash,
        "metadata_sha256": expected_preflight["metadata_sha256"],
        "media_manifest_sha256": expected_preflight["media_manifest_sha256"],
        "audio_checkpoint_sha256": AUDIO_CHECKPOINT_SHA256,
        "audio_source_commit": AUDIO_SOURCE_COMMIT,
        "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
        "video_source_commit": VIDEO_SOURCE_COMMIT,
        "batch_size": BATCH_SIZE,
        "video_frames": 8,
        "video_window": "fixed 0.7 seconds centered on event midpoint",
        "audio_window": "official 5-second DCASE FOA-to-stereo clip",
        "seed": 17,
        "runner_sha256": {
            name: sha256_file(args.repo_root / "scripts" / name)
            for name in (
                "extract_dcase2025_features.py",
                "dcase2025_media_worker.py",
                "dcase2025_audio_worker.py",
                "dcase2025_videoprism_worker.py",
            )
        },
    }
    progress_path = output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != run_identity or progress.get("status") == "PASS":
            raise ValueError("existing DCASE feature run cannot be resumed under this identity")
        completed_batches = int(progress.get("completed_batches", 0))
    else:
        completed_batches = 0
        write_text_atomic(
            output / "query-manifest.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in query_rows),
        )
        write_json_atomic(output / "preflight.json", expected_preflight)
        write_json_atomic(output / "effective-config.json", run_identity)
        progress = {
            "status": "RUNNING",
            "identity": run_identity,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_queries": len(query_rows),
            "completed_batches": 0,
            "completed_rows": 0,
            "valid_rows": 0,
            "clipped_sample_ids": [],
            "worker_seconds": {"media": 0.0, "audio": 0.0, "video": 0.0},
        }
        write_json_atomic(progress_path, progress)
    if completed_batches > 0 and not processed_mask_all(
        output, completed_batches, total, BATCH_SIZE
    ):
        raise ValueError("resume state claims completed feature batches with missing output rows")

    batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for name, shape, dtype in (
        ("features_audio.npy", (total, 768), np.float32),
        ("features_video.npy", (total, 768), np.float32),
        ("valid_mask.npy", (total,), np.bool_),
        ("processed_mask.npy", (total,), np.bool_),
    ):
        path = output / name
        if not path.exists():
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            if dtype == np.float32:
                array[:] = np.nan
            else:
                array[:] = False
            array.flush()
            del array

    features_audio = np.load(output / "features_audio.npy", mmap_mode="r+")
    features_video = np.load(output / "features_video.npy", mmap_mode="r+")
    valid_mask = np.load(output / "valid_mask.npy", mmap_mode="r+")
    processed_mask = np.load(output / "processed_mask.npy", mmap_mode="r+")
    if any(
        len(array) != total
        for array in (features_audio, features_video, valid_mask, processed_mask)
    ):
        raise ValueError("existing feature arrays do not match the frozen query count")

    current_python = Path(sys.executable)
    media_python = args.media_python.resolve(strict=True)
    video_python = args.video_python.resolve(strict=True)
    audio_env = os.environ.copy()
    audio_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.audio_dependency_root.resolve()),
            str(args.repo_root.resolve() / "src"),
            str(args.repo_root.resolve()),
        ]
    )
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [str(args.repo_root.resolve() / "src"), str(args.repo_root.resolve())]
    )
    media_command = [
        str(media_python),
        str(args.repo_root / "scripts/dcase2025_media_worker.py"),
        "--media-manifest",
        str(args.media_manifest.resolve()),
        "--generator-source",
        str(args.generator_source.resolve()),
    ]
    audio_command = [
        str(current_python),
        str(args.repo_root / "scripts/dcase2025_audio_worker.py"),
        "--source",
        str(args.audio_source.resolve()),
        "--checkpoint",
        str(args.audio_checkpoint.resolve()),
        "--dependency-root",
        str(args.audio_dependency_root.resolve()),
    ]
    video_command = [
        str(video_python),
        str(args.repo_root / "scripts/dcase2025_videoprism_worker.py"),
        "--source",
        str(args.video_source.resolve()),
        "--checkpoint",
        str(args.video_checkpoint.resolve()),
    ]

    workers = []
    started = time.perf_counter()
    try:
        media_worker = start_worker(media_command, worker_env, output / "media-worker.log")
        workers.append(media_worker)
        audio_worker = start_worker(audio_command, audio_env, output / "audio-worker.log")
        workers.append(audio_worker)
        video_worker = start_worker(video_command, worker_env, output / "video-worker.log")
        workers.append(video_worker)
        progress["worker_ready"] = {
            "media": media_worker[2],
            "audio": audio_worker[2],
            "video": video_worker[2],
        }
        progress["last_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json_atomic(progress_path, progress)

        for batch_index in range(completed_batches, batch_count):
            first = batch_index * BATCH_SIZE
            last = min(first + BATCH_SIZE, total)
            batch_rows = query_rows[first:last]
            media_path = scratch / f"batch-{batch_index:05d}.npz"
            audio_path = scratch / f"audio-{batch_index:05d}.npy"
            video_path = scratch / f"video-{batch_index:05d}.npy"
            media_response = send_request(
                media_worker[0],
                {
                    "batch_id": batch_index,
                    "queries": batch_rows,
                    "output_npz": str(media_path),
                },
            )
            audio_response = send_request(
                audio_worker[0],
                {
                    "batch_id": batch_index,
                    "media_npz": str(media_path),
                    "output_npy": str(audio_path),
                },
            )
            video_response = send_request(
                video_worker[0],
                {
                    "batch_id": batch_index,
                    "media_npz": str(media_path),
                    "output_npy": str(video_path),
                },
            )
            batch_ids = [row["sample_id"] for row in batch_rows]
            if (
                audio_response.get("sample_ids") != batch_ids
                or video_response.get("sample_ids") != batch_ids
            ):
                raise ValueError(f"worker row order mismatch in batch {batch_index}")
            with np.load(media_path, allow_pickle=False) as media_arrays:
                valid = media_arrays["valid"]
                if media_arrays["sample_ids"].tolist() != batch_ids:
                    raise ValueError(f"media row order mismatch in batch {batch_index}")
            audio_values = np.load(audio_path, allow_pickle=False)
            video_values = np.load(video_path, allow_pickle=False)
            if audio_values.shape != (len(batch_rows), 768) or video_values.shape != (
                len(batch_rows),
                768,
            ):
                raise ValueError(f"feature shape mismatch in batch {batch_index}")
            if not np.isfinite(audio_values).all() or not np.isfinite(video_values).all():
                raise ValueError(f"non-finite features in batch {batch_index}")

            features_audio[first:last] = audio_values
            features_video[first:last] = video_values
            valid_mask[first:last] = valid
            processed_mask[first:last] = True
            features_audio.flush()
            features_video.flush()
            valid_mask.flush()
            processed_mask.flush()
            progress["completed_batches"] = batch_index + 1
            progress["completed_rows"] = last
            progress["valid_rows"] = int(valid_mask[:last].sum())
            progress["clipped_sample_ids"] = [
                query_rows[index]["sample_id"]
                for index in range(last)
                if processed_mask[index] and not valid_mask[index]
            ]
            progress["worker_seconds"]["media"] += float(media_response["wall_seconds"])
            progress["worker_seconds"]["audio"] += float(audio_response["wall_seconds"])
            progress["worker_seconds"]["video"] += float(video_response["wall_seconds"])
            write_json_atomic(progress_path, progress)
            media_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            if (batch_index + 1) % 4 == 0 or batch_index + 1 == batch_count:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "batches": f"{batch_index + 1}/{batch_count}",
                            "processed": last,
                            "valid": progress["valid_rows"],
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                            "mean_seconds_per_clip": round(
                                (time.perf_counter() - started) / last, 3
                            ),
                        }
                    ),
                    flush=True,
                )

        if not processed_mask.all():
            raise ValueError("feature extraction ended before all rows were processed")
        progress.update(
            status="PASS",
            total_wall_seconds=time.perf_counter() - started,
            feature_audio_sha256=sha256_file(output / "features_audio.npy"),
            feature_video_sha256=sha256_file(output / "features_video.npy"),
            processed_mask_sha256=sha256_file(output / "processed_mask.npy"),
            valid_mask_sha256=sha256_file(output / "valid_mask.npy"),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        write_json_atomic(progress_path, progress)
    except Exception as error:
        progress.update(
            status="INTERRUPTED",
            last_error=f"{type(error).__name__}: {error}",
            interrupted_after_batches=progress.get("completed_batches", 0),
        )
        write_json_atomic(progress_path, progress)
        raise
    finally:
        for process, log, _ready in workers:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        for array in (features_audio, features_video, valid_mask, processed_mask):
            del array
    return progress


def zipfile_for(path: Path):
    import zipfile

    return zipfile.ZipFile(path)


def processed_mask_all(output: Path, completed_batches: int, total: int, batch_size: int) -> bool:
    processed = np.load(output / "processed_mask.npy", mmap_mode="r")
    completed_rows = min(completed_batches * batch_size, total)
    valid = len(processed) == total and bool(processed[:completed_rows].all())
    del processed
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--media-python", type=Path, required=True)
    parser.add_argument("--generator-source", type=Path, required=True)
    parser.add_argument("--audio-source", type=Path, required=True)
    parser.add_argument("--audio-checkpoint", type=Path, required=True)
    parser.add_argument("--audio-dependency-root", type=Path, required=True)
    parser.add_argument("--video-python", type=Path, required=True)
    parser.add_argument("--video-source", type=Path, required=True)
    parser.add_argument("--video-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
