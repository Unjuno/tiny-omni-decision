"""Extract frozen causal VideoPrism features from a verified HapTile split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.pretrained_reuse.analyze_haptile_development_preflight import (  # noqa: E402
    validate_development_receipt,
)
from scripts.pretrained_reuse.analyze_haptile_train_preflight import (  # noqa: E402
    validate_train_receipt,
)
from scripts.pretrained_reuse.dextraverse_features import (  # noqa: E402
    read_video_features,
    write_video_batch,
)
from scripts.pretrained_reuse.extract_haptile_development_state_text import (  # noqa: E402
    validate_development_action,
)
from tiny_omni_decision.haptile_video import (  # noqa: E402
    select_causal_frame_indices,
)

VIDEO_TOPICS = (
    "/front-camera",
    "/wrist-camera",
    "/tactile-left",
    "/tactile-right",
)
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
MODEL_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
FEATURE_SHAPE = (4, 8, 768)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_feature_memmap(features: np.memmap, partial_path: Path, final_path: Path) -> None:
    """Flush and close an NPY memmap before renaming it (required on Windows)."""
    features.flush()
    mmap = getattr(features, "_mmap", None)
    if mmap is not None:
        mmap.close()
    os.rename(partial_path, final_path)


def _git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_train_inputs(
    receipt_path: Path,
    action_path: Path,
    split_manifest_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    validated_files = validate_train_receipt(
        receipt,
        split_manifest,
        sha256_file(split_manifest_path),
    )
    action = json.loads(action_path.read_text(encoding="utf-8"))
    if action.get("format") != "haptile_train_only_action_preflight_v1":
        raise ValueError("input action preflight is not a train-only HapTile artifact")
    if action.get("input_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("action targets were derived from a different train receipt")
    if action.get("split_manifest_sha256") != sha256_file(split_manifest_path):
        raise ValueError("action targets use a different frozen task split")
    if action.get("validation_or_sealed_audit_opened") is not False:
        raise ValueError("action preflight does not prove held-out split isolation")
    action_rows = action.get("samples")
    if not isinstance(action_rows, list) or len(action_rows) != len(validated_files):
        raise ValueError("action rows must match every receipt file exactly once")
    rows_by_path = {str(row["path"]): row for row in action_rows}
    if len(rows_by_path) != len(action_rows):
        raise ValueError("action preflight contains duplicate sample paths")
    samples: list[dict[str, object]] = []
    for item in validated_files:
        path = str(item["path"])
        row = rows_by_path.get(path)
        if row is None or row.get("sha256") != item["sha256"]:
            raise ValueError(f"action target and train-media receipt differ: {path}")
        if "query_time_log_seconds" not in row or "target" not in row:
            raise ValueError(f"action row lacks the frozen query/target fields: {path}")
        samples.append({
            "path": path,
            "task": row.get("task", Path(path).parts[1]),
            "sha256": item["sha256"],
            "query_time_log_seconds": float(row["query_time_log_seconds"]),
            "target": str(row["target"]),
        })
    return receipt, samples


def _load_development_inputs(
    receipt_path: Path, action_path: Path, split_manifest_path: Path, split_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    files = validate_development_receipt(
        receipt, manifest, sha256_file(split_manifest_path), split_name
    )
    action = json.loads(action_path.read_text(encoding="utf-8"))
    rows = validate_development_action(
        action=action, receipt=receipt, receipt_path=receipt_path,
        manifest_path=split_manifest_path, split_name=split_name, file_rows=files,
    )
    samples: list[dict[str, object]] = []
    for item in files:
        path = str(item["path"])
        row = rows.get(path)
        if row is None or row.get("sha256") != item["sha256"]:
            raise ValueError(f"action target and development media differ: {path}")
        if "query_time_log_seconds" not in row or "target" not in row:
            raise ValueError(f"development action row lacks query/target: {path}")
        samples.append({
            "path": path,
            "task": item.get("task", Path(path).parts[1]),
            "sha256": item["sha256"],
            "query_time_log_seconds": float(row["query_time_log_seconds"]),
            "target": str(row["target"]),
        })
    return receipt, samples


def _decode_causal_window(
    path: Path, topic: str, query_time_s: float
) -> tuple[np.ndarray, list[float]]:
    try:
        import av
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError(
            "install pinned MCAP/PyAV dependencies in the isolated input environment"
        ) from exc

    codec = av.CodecContext.create("h264", "r")
    codec.thread_count = 1
    timestamps: list[float] = []
    frames: list[np.ndarray] = []
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for schema, channel, message, compressed in reader.iter_decoded_messages(topics=[topic]):
            timestamp = message.log_time / 1_000_000_000
            if timestamp > query_time_s:
                break
            if schema is None or schema.name != "foxglove.CompressedVideo":
                raise ValueError(f"unexpected video schema on {topic}: {path}")
            if channel.topic != topic or compressed.format != "h264":
                raise ValueError(f"unexpected video topic/codec on {topic}: {path}")
            decoded = codec.decode(av.Packet(compressed.data))
            if len(decoded) != 1:
                raise ValueError(
                    f"expected one timestamped H.264 frame per message, got {len(decoded)}"
                )
            if timestamp >= query_time_s - 1.1:
                frame = decoded[0]
                if (frame.width, frame.height) != (320, 240):
                    raise ValueError(
                        f"unexpected HapTile video resolution: {frame.width}x{frame.height}"
                    )
                timestamps.append(timestamp)
                frames.append(frame.to_ndarray(format="rgb24"))
        delayed = codec.decode(None)
        if delayed:
            raise ValueError("H.264 decoder emitted untimestamped frames at flush")
    indices = select_causal_frame_indices(timestamps, query_time_s)
    selected_times = [timestamps[index] for index in indices]
    selected_frames = np.stack([frames[index] for index in indices])
    if selected_frames.shape != (8, 240, 320, 3) or selected_frames.dtype != np.uint8:
        raise ValueError("causal HapTile video window has an invalid frame tensor")
    return selected_frames, selected_times


def _start_worker(
    *,
    worker_python: Path,
    checkpoint: Path,
    source_root: Path,
    report_path: Path,
    stderr_path: Path,
) -> tuple[subprocess.Popen[bytes], object, dict[str, object]]:
    worker_python = worker_python.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 differs from the pinned checkpoint")
    if _git_commit(source_root) != MODEL_SOURCE_COMMIT:
        raise ValueError("VideoPrism source checkout is not the pinned upstream commit")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(REPO_ROOT), str(REPO_ROOT / "src"), existing_pythonpath) if item
    )
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    stderr_stream = stderr_path.open("wb")
    process = subprocess.Popen(
        [
            str(worker_python),
            "-m",
            "scripts.pretrained_reuse.dextraverse_videoprism_worker",
            "--checkpoint",
            str(checkpoint),
            "--source-root",
            str(source_root),
            "--device",
            "cpu",
            "--worker-report",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_stream,
    )
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "ready" or report.get("platform") != "cpu":
                process.terminate()
                process.wait(timeout=30)
                stderr_stream.close()
                raise RuntimeError("VideoPrism worker did not report pinned CPU readiness")
            return process, stderr_stream, report
        if process.poll() is not None:
            stderr_stream.close()
            raise RuntimeError(
                f"VideoPrism worker exited before readiness with code {process.returncode}"
            )
        time.sleep(0.25)
    process.terminate()
    process.wait(timeout=30)
    stderr_stream.close()
    raise TimeoutError("VideoPrism worker did not become ready within 600 seconds")


def extract_features(
    *,
    receipt_path: Path,
    action_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    worker_python: Path,
    checkpoint: Path,
    source_root: Path,
    split_name: str = "train",
    limit: int | None = None,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if split_name == "train":
        _, all_samples = _load_train_inputs(receipt_path, action_path, split_manifest_path)
    elif split_name in {"validation", "calibration"}:
        _, all_samples = _load_development_inputs(
            receipt_path, action_path, split_manifest_path, split_name
        )
    else:
        raise ValueError("split must be train, validation, or calibration")
    samples = all_samples if limit is None else all_samples[:limit]
    if not samples or (limit is not None and limit <= 0):
        raise ValueError("feature extraction requires a positive sample count")
    output_dir.mkdir(parents=True)
    script_copy = output_dir / "extract_haptile_videoprism_features.executed.py"
    shutil.copy2(Path(__file__), script_copy)
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed_examples": 0,
                "total_examples": len(samples),
                "status": "starting",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "worker-report.json"
    worker_log_path = output_dir / "worker-stderr.log"
    process, stderr_stream, worker_report = _start_worker(
        worker_python=worker_python,
        checkpoint=checkpoint,
        source_root=source_root,
        report_path=report_path,
        stderr_path=worker_log_path,
    )
    partial_features_path = output_dir / "video-features.partial.npy"
    features = np.lib.format.open_memmap(
        partial_features_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(samples), *FEATURE_SHAPE),
    )
    examples: list[dict[str, object]] = []
    feature_row_hashes: list[str] = []
    decode_seconds = 0.0
    encoder_seconds = 0.0
    started = time.perf_counter()
    try:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("VideoPrism worker pipes were not created")
        for sample_index, sample in enumerate(samples):
            relative = Path(str(sample["path"]))
            media_path = receipt_path.parent.joinpath(*relative.parts)
            if sha256_file(media_path) != sample["sha256"]:
                raise ValueError(f"HapTile {split_name} media hash changed: {relative}")
            query_time = float(sample["query_time_log_seconds"])
            clips: list[np.ndarray] = []
            selected_timestamps: dict[str, list[float]] = {}
            decode_started = time.perf_counter()
            for topic in VIDEO_TOPICS:
                clip, frame_times = _decode_causal_window(media_path, topic, query_time)
                clips.append(clip)
                selected_timestamps[topic] = frame_times
            decode_seconds += time.perf_counter() - decode_started
            write_video_batch(process.stdin, np.stack(clips))
            encoder_started = time.perf_counter()
            encoded = read_video_features(process.stdout, len(clips))
            encoder_seconds += time.perf_counter() - encoder_started
            if encoded.shape != FEATURE_SHAPE or not np.isfinite(encoded).all():
                raise ValueError("VideoPrism returned invalid temporal features")
            features[sample_index] = encoded
            features.flush()
            feature_row_hashes.append(hashlib.sha256(encoded.tobytes()).hexdigest())
            examples.append({
                "path": sample["path"],
                "task": sample["task"],
                "sha256": sample["sha256"],
                "query_time_log_seconds": query_time,
                "target": sample["target"],
                "selected_frame_times_log_seconds": selected_timestamps,
            })
            progress = {
                "completed_examples": sample_index + 1,
                "total_examples": len(samples),
                "last_path": sample["path"],
                "feature_row_sha256": feature_row_hashes,
                "status": "in_progress",
            }
            (output_dir / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n", encoding="utf-8"
            )
            if (sample_index + 1) % 4 == 0:
                print(
                    f"encoded {sample_index + 1}/{len(samples)} {split_name} examples",
                    flush=True,
                )

        process.stdin.write(struct.pack("<I", 0))
        process.stdin.flush()
        process.stdin.close()
        return_code = process.wait(timeout=120)
        stderr_stream.close()
        if return_code != 0:
            raise RuntimeError(f"VideoPrism worker exited with code {return_code}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        stderr_stream.close()
        raise

    features_path = output_dir / "video-features.npy"
    finalize_feature_memmap(features, partial_features_path, features_path)
    examples_path = output_dir / "examples.json"
    examples_path.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    result: dict[str, object] = {
        "format": f"haptile_{split_name}_videoprism_features_v1",
        "status": "complete_frozen_feature_extraction",
        "model_revision": MODEL_REVISION,
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_parameters": 114_365_184,
        "requested_device": "cpu",
        "worker_report": worker_report,
        "input_receipt_sha256": sha256_file(receipt_path),
        "action_preflight_sha256": sha256_file(action_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "stream_order": list(VIDEO_TOPICS),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "examples": len(examples),
        "clip_count": len(examples) * len(VIDEO_TOPICS),
        "decode_seconds": decode_seconds,
        "encoder_wait_seconds": encoder_seconds,
        "wall_seconds": time.perf_counter() - started,
        "features_sha256": sha256_file(features_path),
        "examples_sha256": sha256_file(examples_path),
        "executed_script_sha256": sha256_file(script_copy),
        "development_split": split_name if split_name != "train" else None,
        "sealed_audit_opened": False,
        "validation_or_sealed_audit_opened": False,
        "audio_present_in_source": False,
        "product_accuracy_claim": False,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed_examples": len(examples),
                "total_examples": len(samples),
                "status": "complete",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--action-preflight", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "calibration"), default="train"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = extract_features(
        receipt_path=args.receipt,
        action_path=args.action_preflight,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        worker_python=args.worker_python,
        checkpoint=args.checkpoint,
        source_root=args.source_root,
        split_name=args.split,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
