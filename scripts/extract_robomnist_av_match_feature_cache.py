"""Stream the frozen RoboMNIST train/validation media into pinned CPU features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import probe_robomnist_av_match_features as probe  # noqa: E402
from tiny_omni_decision.robomnist_media import decode_pinned_member  # noqa: E402

EXPECTED_MANIFEST_SHA256 = probe.SPLIT_SHA256
FEATURE_CONTRACT = {
    "video_frames": 8,
    "video_feature_layout": "per_frame_mean",
    "audio_windows": 8,
    "audio_window_seconds": 2.0,
    "audio_video_alignment": "episode-level only; no common frame clock",
    "precision": "float32",
    "devices": "CPU only",
    "raw_media_persisted": False,
}
MIN_FREE_BYTES = 2 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_order(manifest: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    by_split: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        seen: set[str] = set()
        ordered: list[str] = []
        for unit in manifest["pair_units"][split]:
            for group_id in (unit["group_a"], unit["group_b"]):
                if group_id not in seen:
                    seen.add(group_id)
                    ordered.append(group_id)
        by_split[split] = ordered
    overlap = set(by_split["train"]) & set(by_split["validation"])
    if overlap:
        raise ValueError(
            f"underlying video/audio groups cross train and validation: {sorted(overlap)}"
        )
    return by_split["train"] + by_split["validation"], by_split


def _json_write_atomic(path: Path, value: dict[str, Any]) -> None:
    attempt = 0
    while True:
        temporary = path.with_name(f"{path.name}.partial-{attempt:04d}")
        if not temporary.exists():
            break
        attempt += 1
    with temporary.open("x", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _save_features(path: Path, *, group_id: str, video: np.ndarray, audio: np.ndarray) -> None:
    attempt = 0
    while True:
        temporary = path.with_name(f"{path.name}.partial-{attempt:04d}")
        if not temporary.exists():
            break
        attempt += 1
    with temporary.open("xb") as target:
        np.savez_compressed(
            target,
            group_id=np.asarray(group_id),
            video_features=video.astype(np.float32, copy=False),
            audio_features=audio.astype(np.float32, copy=False),
        )
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _read_receipt(receipt_path: Path, feature_path: Path, group_id: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETE" or receipt.get("group_id") != group_id:
        raise ValueError(f"invalid completion receipt for {group_id}")
    if not feature_path.is_file() or sha256_file(feature_path) != receipt.get("feature_sha256"):
        raise ValueError(f"feature file hash mismatch for {group_id}")
    with np.load(feature_path, allow_pickle=False) as data:
        if str(data["group_id"].item()) != group_id:
            raise ValueError(f"feature group ID mismatch for {group_id}")
        video = data["video_features"]
        audio = data["audio_features"]
    if (
        video.shape != (8, 768)
        or audio.shape != (8, 384)
        or not np.isfinite(video).all()
        or not np.isfinite(audio).all()
    ):
        raise ValueError(f"feature shape/value check failed for {group_id}")
    return receipt


def _new_group_stage(feature_dir: Path, group_id: str) -> Path:
    attempt = 0
    while True:
        stage = feature_dir / f"{group_id}.partial-{attempt:04d}"
        try:
            stage.mkdir()
            return stage
        except FileExistsError:
            attempt += 1


def _extract_group(
    group_id: str,
    source: dict[str, Any],
    local_group_id: str,
    local_payloads: dict[str, bytes],
    *,
    jax: Any,
    tf: Any,
    video_encode: Any,
    audio_model: Any,
    audio_mel: Any,
    torch: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    payloads = local_payloads.copy() if group_id == local_group_id else {}
    media_records: dict[str, Any] = {}
    network_bytes = 0
    for modality, member_key in (("video", "video_member"), ("audio", "audio_member")):
        member = source[member_key]
        if modality in payloads:
            content = payloads[modality]
            range_bytes = 0
            range_seconds = 0.0
        else:
            fetch_started = time.perf_counter()
            header = probe._request_range(
                source["archive_url"], member["offset"], 30, source["archive_size"]
            )
            range_start, tail_length = probe.compressed_member_range(header, member)
            tail = probe._request_range(
                source["archive_url"], range_start, tail_length, source["archive_size"]
            )
            content = decode_pinned_member(header, tail, member)
            range_bytes = 30 + tail_length
            network_bytes += range_bytes
            range_seconds = time.perf_counter() - fetch_started
        payloads[modality] = content
        media_records[modality] = {
            "member": member["name"],
            "size": len(content),
            "sha256": probe.sha256_bytes(content),
            "crc32_expected": member["crc32"],
            "range_bytes": range_bytes,
            "range_seconds": range_seconds,
        }
    frames, video_timing = probe._video_frames(payloads["video"])
    video_features = probe._encode_video(frames, jax=jax, tf=tf, encode=video_encode)
    windows, audio_timing = probe._audio_windows(payloads["audio"])
    audio_features = probe._encode_audio(windows, audio_model, audio_mel, torch)
    receipt = {
        "group_id": group_id,
        "media": media_records,
        "video_timing": video_timing,
        "audio_timing": audio_timing,
    }
    return video_features, audio_features, receipt, network_bytes


def run(
    manifest_path: Path, output: Path, *, resume: bool, max_groups: int | None
) -> dict[str, Any]:
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("frozen attempt-03 manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups, split_groups = group_order(manifest)
    if max_groups is None and len(groups) != manifest["split"]["selected_group_count"]:
        raise ValueError("selected split group count differs from its frozen manifest")
    if max_groups is not None:
        if max_groups < 1:
            raise ValueError("--max-groups must be positive")
        groups = groups[:max_groups]
        split_groups = {
            name: [group_id for group_id in rows if group_id in set(groups)]
            for name, rows in split_groups.items()
        }
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("--resume requires an existing output directory")
    elif output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    else:
        output.mkdir(parents=True)
    feature_dir = output / "groups"
    feature_dir.mkdir(exist_ok=True)
    group_list_hash = hashlib.sha256("\n".join(groups).encode()).hexdigest()
    run_path = output / "run.json"
    effective = {
        "manifest_sha256": manifest_hash,
        "split_group_ids_sha256": manifest["split"]["group_ids_sha256"],
        "group_order_sha256": group_list_hash,
        "group_count": len(groups),
        "split_group_counts": {name: len(rows) for name, rows in split_groups.items()},
        "feature_contract": FEATURE_CONTRACT,
        "video_checkpoint_sha256": probe.VIDEO_SHA256,
        "video_source_commit": probe.VIDEO_SOURCE_COMMIT,
        "audio_checkpoint_sha256": probe.AUDIO_SHA256,
        "audio_source_commit": probe.AUDIO_SOURCE_COMMIT,
        "implementation_sha256": {
            "extractor": sha256_file(Path(__file__).resolve()),
            "probe": sha256_file(Path(probe.__file__).resolve()),
            "media_helper": sha256_file(
                REPO_ROOT / "src" / "tiny_omni_decision" / "robomnist_media.py"
            ),
        },
        "minimum_free_bytes": MIN_FREE_BYTES,
    }
    if resume:
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("effective") != effective:
            raise ValueError("resume contract differs from original frozen run")
        if prior.get("training") is not False or prior.get("sealed_audit_opened") is not False:
            raise ValueError("existing output is not a feature-only development run")
        prior.pop("error", None)
        prior.pop("error_type", None)
    else:
        prior = {
            "status": "RUNNING",
            "created_at": datetime.now(UTC).isoformat(),
            "effective": effective,
            "training": False,
            "quality_evaluated": False,
            "sealed_audit_opened": False,
            "raw_media_persisted": False,
            "completed_groups": 0,
        }
        _json_write_atomic(run_path, prior)

    unexpected_partial = list(output.rglob("*.partial-*"))
    if unexpected_partial:
        # Never discard interrupted output. A later attempt uses a different suffix.
        prior["preserved_partial_files"] = [str(path) for path in unexpected_partial]

    started = time.perf_counter()
    network_bytes_total = 0
    completed = 0
    try:
        sys.path.insert(0, str(probe.REPO_ROOT))
        sys.path.insert(0, str(probe.REPO_ROOT / "src"))
        import jax
        import numpy as vp_numpy
        import tensorflow as tf

        sys.path.append(str(probe.GLOBAL_SITE))
        sys.path.append(str(probe.PYTHON_DEPS))
        import torch
        import torchaudio

        tf.config.set_visible_devices([], "GPU")
        if (
            vp_numpy.__version__ != "1.26.4"
            or np.__version__ != "1.26.4"
            or jax.default_backend() != "cpu"
            or tf.config.get_visible_devices("GPU")
            or torch.cuda.is_initialized()
        ):
            raise ValueError("pinned CPU runtime contract was not satisfied")
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        local_group_id, local_payloads, prior_input_hash = probe._load_local_assets(
            manifest, probe.LOCAL_INPUT_ROOT / "result.json"
        )
        source_pins = probe._pin_sources(prior_input_hash)
        video_model, video_variables, video_encode, video_parameter_count = (
            probe._load_video_encoder(jax, probe.VIDEO_MODEL, probe.VIDEO_SOURCE)
        )
        audio_model, audio_mel, audio_parameter_count = probe._load_audio_encoder(
            torch, torchaudio, probe.AUDIO_WEIGHTS, probe.AUDIO_SOURCE
        )
        del video_model, video_variables
        audio_state_hash = hashlib.sha256(
            b"".join(
                name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
                for name, tensor in sorted(audio_model.state_dict().items())
            )
        ).hexdigest()
        runtime = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": jax.__version__,
            "tensorflow": tf.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "jax_backend": jax.default_backend(),
            "tensorflow_visible_gpus": tf.config.get_visible_devices("GPU"),
            "torch_cuda_initialized": torch.cuda.is_initialized(),
        }
        if resume:
            if prior.get("source_pins") != source_pins or prior.get("runtime") != runtime:
                raise ValueError("resume source pins or runtime differ from the original run")
        prior.update({"source_pins": source_pins, "runtime": runtime, "status": "RUNNING"})
        _json_write_atomic(run_path, prior)

        for index, group_id in enumerate(groups, start=1):
            free_bytes = shutil.disk_usage(output).free
            if free_bytes < MIN_FREE_BYTES:
                raise OSError(f"disk reserve reached: {free_bytes} free bytes < {MIN_FREE_BYTES}")
            group_dir = feature_dir / group_id
            if group_dir.exists():
                _read_receipt(group_dir / "receipt.json", group_dir / "features.npz", group_id)
                completed += 1
                if index % 10 == 0 or index == len(groups):
                    print(
                        json.dumps(
                            {
                                "group": index,
                                "total": len(groups),
                                "id": group_id,
                                "status": "verified-existing",
                            }
                        ),
                        flush=True,
                    )
                continue
            group_started = time.perf_counter()
            video, audio, record, network_bytes = _extract_group(
                group_id,
                manifest["groups"][group_id],
                local_group_id,
                local_payloads,
                jax=jax,
                tf=tf,
                video_encode=video_encode,
                audio_model=audio_model,
                audio_mel=audio_mel,
                torch=torch,
            )
            stage = _new_group_stage(feature_dir, group_id)
            feature_path = stage / "features.npz"
            _save_features(feature_path, group_id=group_id, video=video, audio=audio)
            record.update(
                {
                    "status": "COMPLETE",
                    "feature_sha256": sha256_file(feature_path),
                    "feature_bytes": feature_path.stat().st_size,
                    "elapsed_seconds": time.perf_counter() - group_started,
                }
            )
            _json_write_atomic(stage / "receipt.json", record)
            _read_receipt(stage / "receipt.json", feature_path, group_id)
            os.replace(stage, group_dir)
            completed += 1
            network_bytes_total += network_bytes
            prior.update(
                {
                    "completed_groups": completed,
                    "last_completed_group": group_id,
                    "network_bytes_this_process": network_bytes_total,
                    "elapsed_seconds_this_process": time.perf_counter() - started,
                }
            )
            _json_write_atomic(run_path, prior)
            if index % 10 == 0 or index == len(groups):
                print(
                    json.dumps(
                        {
                            "group": index,
                            "total": len(groups),
                            "id": group_id,
                            "status": "complete",
                            "group_network_bytes": network_bytes,
                            "feature_bytes": record["feature_bytes"],
                            "elapsed_seconds": record["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
        if (
            audio_state_hash
            != hashlib.sha256(
                b"".join(
                    name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
                    for name, tensor in sorted(audio_model.state_dict().items())
                )
            ).hexdigest()
        ):
            raise RuntimeError("frozen audio encoder weights changed during extraction")
        prior.update(
            {
                "status": "COMPLETE",
                "finished_at": datetime.now(UTC).isoformat(),
                "completed_groups": completed,
                "network_bytes_this_process": network_bytes_total,
                "elapsed_seconds_this_process": time.perf_counter() - started,
                "resident_encoder_parameters": video_parameter_count + audio_parameter_count,
                "trainable_parameters": 0,
                "sealed_audit_opened": False,
                "quality_evaluated": False,
            }
        )
        _json_write_atomic(run_path, prior)
        return prior
    except BaseException as exc:
        prior.update(
            {
                "status": "INCOMPLETE",
                "updated_at": datetime.now(UTC).isoformat(),
                "completed_groups": completed,
                "network_bytes_this_process": network_bytes_total,
                "elapsed_seconds_this_process": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "preserved_partial_files": [str(path) for path in output.rglob("*.partial-*")],
            }
        )
        _json_write_atomic(run_path, prior)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.manifest.resolve(strict=True),
                args.output.resolve(),
                resume=args.resume,
                max_groups=args.max_groups,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
