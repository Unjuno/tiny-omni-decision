"""Extract pinned frozen GRID audio and temporal video features for Path A."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from scripts.grid_path_a_data import OPTIONS, validate_direct_split

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SPLIT_DIR = ROOT / "grid-development-split-direct-source-20261028-attempt-02"
BUNDLE_INDEX = ROOT / (
    "grid-train-direct-speaker-audit-20261028-attempt-03/direct-source-bundle-index.json"
)
COHORT = ROOT / "grid-path-a-cohort-20261028-attempt-01/cohort.json"
MEDIA_PREFLIGHT = ROOT / "grid-path-a-cohort-20261028-attempt-01/media-preflight.json"
MEDIA_AUDIT = ROOT / "grid-media-duplicate-audit-20261028-attempt-01/report.json"
SPEECH = Path("C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01")
VIDEO_ENV = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe")
VIDEO_ARCHIVES = ROOT / "grid-video-archives-duplicate-audit-20261028-attempt-01"
S30_VIDEO = ROOT / "grid-validation-video-20261028-attempt-01/s30.zip"
VIDEO_WORKER = Path("scripts/native_av_video_worker.py")
VIDEO_SOURCE = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
VIDEO_REVISION = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEO_CHECKPOINT_SHA = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
COHORT_SHA = "61b70af8cb826ded36986aaa53b55ef88f3e5973542d9237df1c951086bab752"
MEDIA_PREFLIGHT_SHA = "26b06fd0de2463fcbb0a4235db9f378c85aad8ec990c6c521c4416dff8aaa24a"
MEDIA_AUDIT_SHA = "a0a756bcf49baa4939dd51bcb4e89f02c4b8b36b891e4b8415fa577ab798712e"
BUNDLE_INDEX_SHA = "fb54bb49a0f9425cd6206508e99a1d2fef25220444e3853af11aac75431861ed"
AUDIO_ENCODER_SHA = "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8"
AUDIO_CONFIG_SHA = "d8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9"
AUDIO_PROCESSOR_SHA = "3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef"
AUDIO_DIM = 384
VIDEO_SHAPE = (16, 768)
WORKER_CHUNK_SIZE = 150


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_audio_member(archive: tarfile.TarFile, speaker: str, code: str) -> bytes:
    member = archive.extractfile(f"{speaker}/{code}.wav")
    if member is None:
        raise ValueError(f"missing direct-source audio member: {speaker}/{code}.wav")
    data = member.read()
    if not data:
        raise ValueError("empty audio member")
    return data


def decode_audio(data: bytes, processor: object) -> tuple[object, float]:
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        if stream.codec_context.sample_rate != 25000:
            raise ValueError("direct GRID endpointed audio is not 25 kHz")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(value.to_ndarray() for value in resampler.resample(frame))
        chunks.extend(value.to_ndarray() for value in resampler.resample(None))
    wave = np.concatenate(chunks, axis=1)[0]
    if not 0 < len(wave) <= 480000 or not np.isfinite(wave).all():
        raise ValueError("invalid resampled waveform")
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt").input_features
    return inputs, len(wave) / 16000


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cohort_rows(
    cohort: dict, train_rows: list[dict], validation_rows: list[dict], smoke: bool
):
    if cohort.get("status") != "PASS_FROZEN_GROUPED_COHORT":
        raise ValueError("frozen Path A cohort receipt did not pass")
    split = validate_direct_split(train_rows, validation_rows)
    by_id = {
        f"{row['speaker']}_{row['code']}": row for row in train_rows + validation_rows
    }
    train_ids = cohort["train_unique_assets"]
    validation_ids = cohort["validation_unique_assets"]
    if (
        len(train_ids) != len(set(train_ids))
        or len(validation_ids) != len(set(validation_ids))
        or set(train_ids) & set(validation_ids)
        or set(validation_ids) & set(cohort["excluded_prior_validation_media"])
    ):
        raise ValueError("cohort contains duplicate, cross-split or previously observed media")
    if any(asset not in by_id for asset in train_ids + validation_ids):
        raise ValueError("cohort contains an asset outside the frozen direct-source split")
    actual_train = set(split.train_ids)
    actual_validation = set(split.validation_ids)
    if not set(train_ids) <= actual_train or not set(validation_ids) <= actual_validation:
        raise ValueError("cohort train/validation asset membership differs")
    if len(train_ids) != 768 or len(validation_ids) != 256:
        raise ValueError("frozen 768/256 Path A cohort required")
    train_order = [(asset, by_id[asset]) for asset in train_ids]
    validation_order = [(asset, by_id[asset]) for asset in validation_ids]
    if smoke:
        train_order, validation_order = train_order[:2], validation_order[:2]
    return train_order + validation_order, len(train_order), len(validation_order)


def _source_config(
    cohort: dict,
    order: list[tuple[str, dict]],
    train_rows: int,
    validation_rows: int,
    smoke: bool,
) -> dict:
    expected = {
        "train.json": "3a4304e7158ee89380a6d4827ef23c5f1117494e10b1493bcb9a97a3fe3be816",
        "validation.json": "f05cf77258aa2ea2145eb0002fc828b9f2cdf7b5f7d0f91adea72b3204a70812",
    }
    if any(sha256_file(SPLIT_DIR / name) != digest for name, digest in expected.items()):
        raise ValueError("corrected frozen GRID split hash differs")
    for path, digest in (
        (COHORT, COHORT_SHA),
        (MEDIA_PREFLIGHT, MEDIA_PREFLIGHT_SHA),
        (MEDIA_AUDIT, MEDIA_AUDIT_SHA),
        (BUNDLE_INDEX, BUNDLE_INDEX_SHA),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"frozen GRID source receipt changed: {path.name}")
    preflight = _read_json(MEDIA_PREFLIGHT)
    if preflight.get("status") != "PASS_SELECTED_MEDIA_MEMBERS_EXIST":
        raise ValueError("selected media preflight did not pass")
    checkpoint = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz")
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA:
        raise ValueError("pinned VideoPrism checkpoint hash differs")
    audio_pins = {
        "encoder.safetensors": AUDIO_ENCODER_SHA,
        "config.json": AUDIO_CONFIG_SHA,
        "preprocessor_config.json": AUDIO_PROCESSOR_SHA,
    }
    if any(sha256_file(SPEECH / name) != digest for name, digest in audio_pins.items()):
        raise ValueError("pinned Whisper encoder or processor hash differs")
    if subprocess.check_output(
        ["git", "-C", str(VIDEO_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip() != VIDEO_REVISION:
        raise ValueError("pinned VideoPrism source revision differs")
    repo = Path(__file__).resolve().parents[1]
    return {
        "protocol": "grid-path-a-speech-slot-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "extractor_sha256": sha256_file(Path(__file__)),
        "data_contract_sha256": sha256_file(repo / "scripts/grid_path_a_data.py"),
        "cohort_sha256": sha256_file(COHORT),
        "train_sha256": expected["train.json"],
        "validation_sha256": expected["validation.json"],
        "bundle_index_sha256": sha256_file(BUNDLE_INDEX),
        "media_preflight_sha256": sha256_file(MEDIA_PREFLIGHT),
        "media_audit_sha256": sha256_file(MEDIA_AUDIT),
        "video_revision": VIDEO_REVISION,
        "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA,
        "video_features": list(VIDEO_SHAPE),
        "video_backend": "pinned native JAX/TF CPU worker",
        "video_worker_sha256": sha256_file(VIDEO_WORKER),
        "audio_encoder_sha256": AUDIO_ENCODER_SHA,
        "audio_config_sha256": AUDIO_CONFIG_SHA,
        "audio_processor_sha256": AUDIO_PROCESSOR_SHA,
        "audio_feature_dimension": AUDIO_DIM,
        "audio_encoder_parameters": 8_208_384,
        "video_encoder_parameters": 114_365_184,
        "audio_backend": "Torch CPU, frozen Whisper encoder",
        "ordered_asset_ids_sha256": sha256_bytes(
            "\n".join(asset for asset, _ in order).encode()
        ),
        "rows": len(order),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "smoke": smoke,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "transformers", "av", "safetensors")
            },
        },
        "video_worker_chunk_size": WORKER_CHUNK_SIZE,
        "gpu_used": False,
        "sealed_audit_opened": False,
    }


def _prepare_output(output: Path, config: dict, resume: bool):
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("feature output must be external to the repository")
    config_path = output / "effective-config.json"
    progress_path = output / "progress.jsonl"
    audio_path = output / "audio-features.npy"
    video_path = output / "video-features.npy"
    targets_path = output / "targets.npy"
    shape_audio = (config["rows"], AUDIO_DIM)
    shape_video = (config["rows"], *VIDEO_SHAPE)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for path, shape in ((audio_path, shape_audio), (video_path, shape_video)):
            np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape).flush()
        if progress_path.exists() or targets_path.exists():
            raise FileExistsError("new feature output unexpectedly contains progress files")
        return 0
    if not output.is_dir() or _read_json(config_path) != config:
        raise ValueError("resume output missing or effective config differs")
    if (output / "result.json").is_file():
        raise ValueError("completed feature output is immutable; create a new attempt")
    if not audio_path.is_file() or not video_path.is_file():
        raise ValueError("resume feature banks are missing")
    if np.load(audio_path, mmap_mode="r", allow_pickle=False).shape != shape_audio:
        raise ValueError("resume audio feature shape differs")
    if np.load(video_path, mmap_mode="r", allow_pickle=False).shape != shape_video:
        raise ValueError("resume video feature shape differs")
    records = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    if len(records) > config["rows"]:
        raise ValueError("resume progress exceeds cohort length")
    for i, record in enumerate(records):
        if record["index"] != i:
            raise ValueError("resume progress is not contiguous")
    return len(records)


def _init_audio():
    import torch
    from safetensors.torch import load_file
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("feature extraction must not initialize CUDA")
    encoder = WhisperEncoder(WhisperConfig.from_pretrained(SPEECH, local_files_only=True))
    encoder.load_state_dict(load_file(SPEECH / "encoder.safetensors"), strict=True)
    encoder.eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in encoder.parameters()) != 8_208_384:
        raise ValueError("Whisper encoder parameter count differs")
    processor = WhisperFeatureExtractor.from_pretrained(SPEECH, local_files_only=True)
    return torch, encoder, processor


def _audio_feature(torch, encoder, processor, payload: bytes) -> tuple[np.ndarray, str]:
    inputs, duration = decode_audio(payload, processor)
    with torch.inference_mode():
        hidden = encoder(inputs).last_hidden_state[0]
    valid = max(1, min(hidden.shape[0], int((duration * 16000 + 319) // 320)))
    feature = hidden[:valid].mean(0).cpu().numpy().astype(np.float32)
    if feature.shape != (AUDIO_DIM,) or not np.isfinite(feature).all():
        raise ValueError("invalid frozen Whisper feature")
    return feature, sha256_bytes(feature.tobytes())


def _video_archive(speaker: str) -> Path:
    return S30_VIDEO if speaker == "s30" else VIDEO_ARCHIVES / f"{speaker}.zip"


def _run_video_chunk(
    output: Path,
    chunk_number: int,
    items: list[tuple[int, str, dict]],
    video_bank: np.ndarray,
    progress,
    audio_bank: np.ndarray,
    audio_runtime,
    run_state_path: Path,
    total_rows: int,
):
    worker_output = output / f"video-worker-{chunk_number:04d}"
    temp_dir = output / f"temporary-media-{chunk_number:04d}"
    temp_dir.mkdir()
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join((str(repo), str(repo / "src")))
    env["JAX_PLATFORMS"] = "cpu"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    command = [str(VIDEO_ENV), "-m", "scripts.native_av_video_worker",
               "--output", str(worker_output)]
    bundle = _read_json(BUNDLE_INDEX)["speakers"]
    speakers = sorted({row["speaker"] for _, _, row in items})
    tar_handles = {speaker: tarfile.open(bundle[speaker]["audio_tar"], "r:*")
                   for speaker in speakers}
    zip_handles = {speaker: zipfile.ZipFile(_video_archive(speaker)) for speaker in speakers}
    log_path = output / f"video-worker-{chunk_number:04d}.stderr.log"
    temp_paths = []
    p = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            p = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=log, text=True,
                                 encoding="utf-8")
            ready_line = p.stdout.readline()
            if not ready_line or not json.loads(ready_line).get("ready"):
                raise RuntimeError("native VideoPrism worker did not become ready")
            for index, asset, row in items:
                speaker, code = row["speaker"], row["code"]
                audio_bytes = read_audio_member(tar_handles[speaker], speaker, code)
                audio, audio_hash = _audio_feature(*audio_runtime, audio_bytes)
                audio_media_hash = sha256_bytes(audio_bytes)
                audio_bank[index] = audio
                video_path = temp_dir / f"{index:04d}-{asset}.mpg"
                member_name = f"{speaker}/{code}.mpg"
                with zip_handles[speaker].open(member_name) as source:
                    with video_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                video_hash = sha256_file(video_path)
                temp_paths.append(video_path)
                request = {"path": str(video_path), "sha256": video_hash}
                p.stdin.write(json.dumps(request) + "\n")
                p.stdin.flush()
                response_line = p.stdout.readline()
                if not response_line:
                    raise RuntimeError("VideoPrism worker exited before returning a feature")
                response = json.loads(response_line)
                if "error" in response:
                    raise RuntimeError(response["error"])
                video = np.asarray(response["feature"], dtype=np.float32)
                if video.shape != VIDEO_SHAPE or not np.isfinite(video).all():
                    raise ValueError("invalid VideoPrism temporal feature")
                video_bank[index] = video
                audio_bank.flush()
                video_bank.flush()
                progress.write(json.dumps({
                    "index": index,
                    "asset_id": asset,
                    "audio_media_sha256": audio_media_hash,
                    "audio_feature_sha256": audio_hash,
                    "video_media_sha256": video_hash,
                    "video_feature_sha256": sha256_bytes(video.tobytes()),
                    "video_seconds": response["metadata"]["forward_seconds"],
                    "decode_preprocess_seconds": response["metadata"]["decode_preprocess_seconds"],
                }) + "\n")
                progress.flush()
                completed = index + 1
                if completed % 32 == 0 or completed == total_rows:
                    run_state_path.write_text(
                        json.dumps({"status": "RUNNING", "completed_rows": completed,
                                    "total_rows": total_rows, "asset_id": asset}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(json.dumps({"completed": completed, "total": total_rows,
                                      "asset_id": asset}), flush=True)
            p.stdin.close()
            if p.wait(timeout=120) != 0:
                raise RuntimeError("native VideoPrism worker failed terminal hash checks")
        for path in temp_paths:
            path.unlink()
        temp_dir.rmdir()
    finally:
        for archive in tar_handles.values():
            archive.close()
        for archive in zip_handles.values():
            archive.close()
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.perf_counter()
    try:
        cohort = _read_json(COHORT)
        train_rows = _read_json(SPLIT_DIR / "train.json")
        validation_rows = _read_json(SPLIT_DIR / "validation.json")
        order, train_count, validation_count = _load_cohort_rows(
            cohort, train_rows, validation_rows, args.smoke
        )
        config = _source_config(cohort, order, train_count, validation_count, args.smoke)
        start_index = _prepare_output(output, config, args.resume)
        expected_targets = np.asarray([
            [OPTIONS[slot].index(row["targets"][slot]) for slot in OPTIONS]
            for _, row in order
        ], dtype=np.int64)
        target_path = output / "targets.npy"
        if target_path.exists():
            if not args.resume or not np.array_equal(
                np.load(target_path, allow_pickle=False), expected_targets
            ):
                raise ValueError("existing target bank differs from fixed cohort")
        else:
            np.save(target_path, expected_targets)
        ids_path = output / "feature-ids.json"
        expected_ids = [asset for asset, _ in order]
        if ids_path.exists():
            if not args.resume or _read_json(ids_path) != expected_ids:
                raise ValueError("existing feature IDs differ from fixed cohort")
        else:
            ids_path.write_text(json.dumps(expected_ids, indent=2) + "\n", encoding="utf-8")
        audio_bank = np.load(output / "audio-features.npy", mmap_mode="r+", allow_pickle=False)
        video_bank = np.load(output / "video-features.npy", mmap_mode="r+", allow_pickle=False)
        records_path = output / "progress.jsonl"
        prior = [json.loads(line) for line in records_path.read_text().splitlines() if line] \
            if records_path.exists() else []
        for i, record in enumerate(prior):
            if record["asset_id"] != order[i][0]:
                raise ValueError("resume progress asset order differs from frozen cohort")
            if sha256_bytes(audio_bank[i].tobytes()) != record["audio_feature_sha256"]:
                raise ValueError("resume audio feature hash differs")
            if sha256_bytes(video_bank[i].tobytes()) != record["video_feature_sha256"]:
                raise ValueError("resume video feature hash differs")
        if len(prior) != start_index:
            raise ValueError("resume progress and manifest state disagree")
        audio_runtime = _init_audio()
        chunk_number = len(list(output.glob("video-worker-[0-9][0-9][0-9][0-9]")))
        state_path = output / "run-state.json"
        state_path.write_text(
            json.dumps({"status": "RUNNING", "completed_rows": start_index,
                        "total_rows": len(order)}, indent=2) + "\n",
            encoding="utf-8",
        )
        with records_path.open("a", encoding="utf-8") as progress:
            for start in range(start_index, len(order), WORKER_CHUNK_SIZE):
                chunk = [(i, order[i][0], order[i][1])
                         for i in range(start, min(start + WORKER_CHUNK_SIZE, len(order)))]
                _run_video_chunk(output, chunk_number, chunk, video_bank, progress,
                                 audio_bank, audio_runtime, state_path, len(order))
                audio_bank.flush()
                video_bank.flush()
                chunk_number += 1
                if shutil.disk_usage(output).free < 2**30:
                    raise RuntimeError("1 GiB free-disk safety reserve exhausted")
                output_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if output_bytes > 512 * 2**20:
                    raise RuntimeError("Path A feature output exceeded 512 MiB guard")
        if not np.isfinite(audio_bank).all() or not np.isfinite(video_bank).all():
            raise ValueError("feature bank contains unwritten or nonfinite values")
        report = {
            "status": "PASS_GRID_PATH_A_FROZEN_FEATURES",
            "effective_config_sha256": sha256_file(output / "effective-config.json"),
            "audio_features_sha256": sha256_file(output / "audio-features.npy"),
            "video_features_sha256": sha256_file(output / "video-features.npy"),
            "targets_sha256": sha256_file(output / "targets.npy"),
            "progress_sha256": sha256_file(records_path),
            "feature_ids_sha256": sha256_file(ids_path),
            "encoder_parameters": {
                "audio": 8_208_384,
                "video": 114_365_184,
            },
            "video_worker_receipts": [
                {
                    "directory": path.name,
                    "report_sha256": sha256_file(path / "result.json"),
                    "status": _read_json(path / "result.json").get("status"),
                    "calls": len(_read_json(path / "result.json").get("calls", [])),
                    "parameters": _read_json(path / "result.json").get("parameters"),
                    "parameter_sha_before": _read_json(path / "result.json").get(
                        "parameter_sha_before"
                    ),
                    "parameter_sha_after": _read_json(path / "result.json").get(
                        "parameter_sha_after"
                    ),
                }
                for path in sorted(output.glob("video-worker-[0-9][0-9][0-9][0-9]"))
            ],
            "train_rows": train_count,
            "validation_rows": validation_count,
            "audio_shape": list(audio_bank.shape),
            "video_shape": list(video_bank.shape),
            "duration_seconds": time.perf_counter() - started,
            "sealed_audit_opened": False,
            "gpu_used": False,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
        state_path.write_text(
            json.dumps({"status": report["status"], "completed_rows": len(order),
                        "total_rows": len(order),
                        "result_sha256": sha256_file(output / "result.json")},
                       indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report), flush=True)
    except Exception as error:
        state = {"status": "FAILED_INCOMPLETE", "error": repr(error),
                 "traceback": traceback.format_exc(),
                 "elapsed_seconds": time.perf_counter() - started,
                 "sealed_audit_opened": False, "gpu_used": False}
        output.mkdir(parents=True, exist_ok=True)
        (output / "run-state.json").write_text(json.dumps(state, indent=2) + "\n",
                                               encoding="utf-8")
        raise
    finally:
        if "audio_runtime" in locals():
            import torch

            if torch.cuda.is_initialized():
                raise RuntimeError("frozen feature extraction unexpectedly initialized CUDA")
if __name__ == "__main__":
    main()
