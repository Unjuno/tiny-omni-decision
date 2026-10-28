"""Run one fixed ADL training row through pinned frozen audio and sensor encoders."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import numbers
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/"
    "adl-audio-vibration-archive-20261027-attempt-01/23_take-from-stove.zip"
)
ARCHIVE_SHA256 = "04bf32f55c93f54c49525d2b0581cb51f14c1ec302afb56b70a995fa4aa8f706"
MEMBER = "23_take-from-stove/23_take-from-stove_multisensor-1.hdf5"
MEMBER_SHA256 = "73aae1893b07c23c0814362f906e9795dea9a03c7a6c13ab03e2f07e5af5f645"
ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1")
EXTERNAL_DEPS = ROOT / "python-deps"
EXTERNAL_SITE = ROOT / "videoprism-base-r1/venv/Lib/site-packages"
EFFICIENTAT_SOURCE = ROOT / "efficientat-audit-v1/source"
EFFICIENTAT_WEIGHTS = ROOT / "efficientat-audit-v1/dymn04_as.pt"
EFFICIENTAT_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_PARAMETER_COUNT = 1_961_239
EFFICIENTAT_WEIGHTS_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
MOMENT_ROOT = ROOT / "moment-small-sensor-r1-attempt-01"
MOMENT_SOURCE = MOMENT_ROOT / "moment-source"
MOMENT_SOURCE_COMMIT = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
MOMENT_MODEL = MOMENT_ROOT / "model"
MOMENT_WEIGHTS = MOMENT_MODEL / "model.safetensors"
MOMENT_WEIGHTS_SHA256 = "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"
MOMENT_CONFIG_SHA256 = "6456b8e6f017323e8ef372c338c993380e246cb6291ffb8ca21fe7dfc53d0a6f"
MOMENT_PARAMETER_COUNT = 35_337_408
PARTICIPANT_SPLIT_PROTOCOL_SHA256 = (
    "9c301d4a3acc37711d25943e7165533410bb3297630eef19c7f6431d2daaf5f8"
)
FEATURE_PROTOCOL = Path(
    "docs/pretrained_reuse/R2_ADL_AUDIO_VIBRATION_FEATURE_RUNTIME_PROTOCOL_20261028.md"
)
SPLIT_PROTOCOL = Path("docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md")
FEATURE_PROTOCOL_SHA256 = "6851fe75ab890838c90607a399dcfc388ad63c88ecc76722f9e1ac3089a14bbb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def participant_split(participant: int) -> str | None:
    """Return the preregistered ADL split; ID 999 is kept unassigned."""
    if isinstance(participant, bool) or not isinstance(participant, numbers.Integral):
        raise ValueError("participant ID must be an integer")
    participant = int(participant)
    if participant < 0:
        raise ValueError("negative participant ID")
    if participant == 999:
        return None
    digest = hashlib.sha256(f"adl-participant-v1:17:{participant}".encode()).digest()
    return "train" if int.from_bytes(digest[:8], "big") % 100 < 70 else "validation"


def resample_vibration_axes(
    axes: Sequence[np.ndarray],
    sample_rate_hz: float,
    *,
    duration_s: float = 2.0,
    target_samples: int = 512,
) -> np.ndarray:
    """Linearly resample the same initial window of three axes for a runtime smoke."""
    if len(axes) != 3:
        raise ValueError("exactly three vibration axes are required")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample rate must be finite and positive")
    if not np.isfinite(duration_s) or duration_s <= 0 or target_samples <= 0:
        raise ValueError("duration and target sample count must be positive")
    native_samples = int(round(sample_rate_hz * duration_s))
    if native_samples < 2:
        raise ValueError("source window is too short")
    native_time = np.arange(native_samples, dtype=np.float64) / sample_rate_hz
    target_time = np.arange(target_samples, dtype=np.float64) * duration_s / target_samples
    result = []
    for axis in axes:
        values = np.asarray(axis)
        if values.ndim != 1 or len(values) < native_samples:
            raise ValueError("each vibration axis must cover the complete source window")
        values = values[:native_samples].astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("vibration window contains nonfinite values")
        result.append(np.interp(target_time, native_time, values))
    sampled = np.stack(result).astype(np.float32)
    if sampled.shape != (3, target_samples) or not np.isfinite(sampled).all():
        raise ValueError("resampled vibration shape or values are invalid")
    return sampled


def state_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def peak_working_set_bytes() -> int | None:
    """Read this process's peak working set when psutil is available."""
    try:
        import psutil

        process_memory = psutil.Process().memory_info()
    except (ImportError, OSError):
        return None
    return int(getattr(process_memory, "peak_wset", process_memory.rss))


def psutil_version() -> str | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.__version__


def _append_unique(path: Path) -> None:
    resolved = str(path.resolve(strict=True))
    if resolved not in sys.path:
        sys.path.append(resolved)


def _source_commit(path: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError(f"pinned source checkout is dirty: {path}")
    return commit


def _read_training_row(h5py: Any) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".hdf5"))
        if MEMBER not in names or len(names) != 5:
            raise ValueError("expected the five fixed sensor members and sensor-1 member")
        payload = archive.read(MEMBER)
    if sha256_bytes(payload) != MEMBER_SHA256:
        raise ValueError("sensor-1 HDF5 member SHA-256 mismatch")
    with h5py.File(io.BytesIO(payload), "r") as handle:
        participants = handle["participant"][:]
        row = next(
            index
            for index, participant in enumerate(participants)
            if participant_split(participant) == "train"
        )
        participant = int(participants[row])
        trial = int(handle["trial"][row])
        audio = np.asarray(handle["audio"][row])
        axes = [np.asarray(handle[f"vibration-{axis}"][row]) for axis in "xyz"]
        sample_rate_hz = float(handle["vibration_sample_rate"][row])
        timestamp = float(handle["timestamp"][row])
    if participant_split(participant) != "train":
        raise ValueError("selected participant is not assigned to train")
    if audio.dtype != np.int16 or audio.ndim != 1 or len(audio) < 32_000:
        raise ValueError("audio must provide a two-second int16 window at declared 16 kHz")
    sensor = resample_vibration_axes(axes, sample_rate_hz)
    record = {
        "row_index": row,
        "participant_id": participant,
        "trial_id": trial,
        "split": "train",
        "member": MEMBER,
        "member_sha256": MEMBER_SHA256,
        "audio_samples_used": 32_000,
        "audio_declared_sample_rate_hz": 16_000,
        "audio_duration_s": 2.0,
        "vibration_native_axis_lengths": [len(axis) for axis in axes],
        "vibration_source_sample_rate_hz": sample_rate_hz,
        "vibration_duration_s": min(len(axis) / sample_rate_hz for axis in axes),
        "vibration_target_shape": list(sensor.shape),
        "vibration_resampling": "numpy.interp linear interpolation to 512 points over first 2 s",
        "source_timestamp_native_units": timestamp,
    }
    return {**record, "audio": audio[:32_000], "sensor": sensor}, payload


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    for path in (ARCHIVE, EFFICIENTAT_WEIGHTS, MOMENT_WEIGHTS, MOMENT_MODEL / "config.json"):
        path.resolve(strict=True)
    feature_protocol = FEATURE_PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    split_protocol = SPLIT_PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    if sha256_bytes(feature_protocol) != FEATURE_PROTOCOL_SHA256:
        raise ValueError("feature runtime protocol hash mismatch")
    if sha256_bytes(split_protocol) != PARTICIPANT_SPLIT_PROTOCOL_SHA256:
        raise ValueError("participant split protocol hash mismatch")
    if sha256_file(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 mismatch")
    if sha256_file(EFFICIENTAT_WEIGHTS) != EFFICIENTAT_WEIGHTS_SHA256:
        raise ValueError("EfficientAT checkpoint SHA-256 mismatch")
    if sha256_file(MOMENT_WEIGHTS) != MOMENT_WEIGHTS_SHA256:
        raise ValueError("MOMENT checkpoint SHA-256 mismatch")
    if sha256_file(MOMENT_MODEL / "config.json") != MOMENT_CONFIG_SHA256:
        raise ValueError("MOMENT config SHA-256 mismatch")
    efficientat_commit = _source_commit(EFFICIENTAT_SOURCE)
    moment_commit = _source_commit(MOMENT_SOURCE)
    if efficientat_commit != EFFICIENTAT_SOURCE_COMMIT or moment_commit != MOMENT_SOURCE_COMMIT:
        raise ValueError("source revision mismatch")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _append_unique(EXTERNAL_SITE)
    _append_unique(EXTERNAL_DEPS)
    started_wall = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    output.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "started_utc": started_wall,
        "feature_protocol_sha256_lf": FEATURE_PROTOCOL_SHA256,
        "participant_split_protocol_sha256_lf": PARTICIPANT_SPLIT_PROTOCOL_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "efficientat_source_commit": efficientat_commit,
        "efficientat_weights_sha256": EFFICIENTAT_WEIGHTS_SHA256,
        "moment_source_commit": moment_commit,
        "moment_weights_sha256": MOMENT_WEIGHTS_SHA256,
        "moment_config_sha256": MOMENT_CONFIG_SHA256,
        "training": False,
        "validation_waveform_read": False,
        "sealed_audit_opened": False,
        "predictions_or_quality_metrics": False,
        "device": "cpu",
        "cuda_visible_devices": "",
    }
    try:
        import h5py
        import torch
        import torchaudio

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        row, member_payload = _read_training_row(h5py)
        audio = row.pop("audio")
        sensor = row.pop("sensor")
        receipt.update(row)
        audio_started = time.perf_counter()
        sys.path.insert(0, str(EFFICIENTAT_SOURCE))
        with contextlib.redirect_stdout(io.StringIO()):
            from models.dymn.model import get_model
            from models.preprocess import AugmentMelSTFT

        with contextlib.redirect_stdout(io.StringIO()):
            audio_model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        audio_model.load_state_dict(
            torch.load(EFFICIENTAT_WEIGHTS, map_location="cpu", weights_only=True),
            strict=True,
        )
        audio_model.eval().requires_grad_(False)
        mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False)
        audio_state_before = state_sha(audio_model)
        waveform = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, orig_freq=16_000, new_freq=32_000)
        with torch.inference_mode():
            _, audio_embedding = audio_model(mel(waveform).unsqueeze(1))
        audio_features = audio_embedding.float().cpu().numpy()
        audio_state_after = state_sha(audio_model)
        audio_parameters = sum(parameter.numel() for parameter in audio_model.parameters())
        if (
            audio_parameters != EFFICIENTAT_PARAMETER_COUNT
            or
            audio_features.shape != (1, 384)
            or not np.isfinite(audio_features).all()
            or np.linalg.norm(audio_features) <= 0
            or audio_state_before != audio_state_after
            or any(parameter.requires_grad for parameter in audio_model.parameters())
        ):
            raise ValueError("frozen EfficientAT feature smoke failed")
        np.save(output / "audio_features.npy", audio_features, allow_pickle=False)
        receipt["audio"] = {
            "output_shape": list(audio_features.shape),
            "output_sha256": sha256_file(output / "audio_features.npy"),
            "feature_norm": float(np.linalg.norm(audio_features)),
            "state_sha256_before": audio_state_before,
            "state_sha256_after": audio_state_after,
            "parameter_count": audio_parameters,
            "dtype": str(audio_features.dtype),
            "wall_seconds": time.perf_counter() - audio_started,
        }
        del audio_model, mel, waveform, audio_embedding, audio_features
        gc.collect()

        sensor_started = time.perf_counter()
        sys.path.insert(0, str(MOMENT_SOURCE))
        from momentfm import MOMENTPipeline

        sensor_model = MOMENTPipeline.from_pretrained(
            str(MOMENT_MODEL), model_kwargs={"task_name": "embedding"}, local_files_only=True
        )
        sensor_model.init()
        sensor_model.to("cpu").eval().requires_grad_(False)
        sensor_state_before = state_sha(sensor_model)
        sensor_input = torch.from_numpy(sensor).unsqueeze(0).float()
        sensor_mask = torch.ones((1, sensor.shape[-1]), dtype=torch.bool)
        with torch.inference_mode():
            sensor_output = sensor_model(
                x_enc=sensor_input, input_mask=sensor_mask, reduction="none"
            )
            sensor_pooled = sensor_model(
                x_enc=sensor_input, input_mask=sensor_mask, reduction="mean"
            )
        sensor_features = sensor_output.embeddings.float().cpu().numpy()
        sensor_mean_features = sensor_pooled.embeddings.float().cpu().numpy()
        sensor_state_after = state_sha(sensor_model)
        sensor_parameters = sum(parameter.numel() for parameter in sensor_model.parameters())
        if (
            sensor_parameters != MOMENT_PARAMETER_COUNT
            or
            sensor_features.shape != (1, 3, 64, 512)
            or sensor_mean_features.shape != (1, 512)
            or not np.isfinite(sensor_features).all()
            or not np.isfinite(sensor_mean_features).all()
            or np.linalg.norm(sensor_mean_features) <= 0
            or sensor_state_before != sensor_state_after
            or any(parameter.requires_grad for parameter in sensor_model.parameters())
        ):
            raise ValueError("frozen MOMENT feature smoke failed")
        np.save(output / "sensor_patch_features.npy", sensor_features, allow_pickle=False)
        np.save(output / "sensor_mean_features.npy", sensor_mean_features, allow_pickle=False)
        receipt["sensor"] = {
            "input_shape": list(sensor_input.shape),
            "patch_output_shape": list(sensor_features.shape),
            "mean_output_shape": list(sensor_mean_features.shape),
            "patch_output_sha256": sha256_file(output / "sensor_patch_features.npy"),
            "mean_output_sha256": sha256_file(output / "sensor_mean_features.npy"),
            "feature_norm": float(np.linalg.norm(sensor_mean_features)),
            "state_sha256_before": sensor_state_before,
            "state_sha256_after": sensor_state_after,
            "parameter_count": sensor_parameters,
            "dtype": str(sensor_features.dtype),
            "wall_seconds": time.perf_counter() - sensor_started,
        }
        del sensor_model, sensor_input, sensor_mask, sensor_output, sensor_pooled
        del sensor_features, sensor_mean_features
        gc.collect()
        receipt.update(
            {
                "status": "PASS_TRAIN_ROW_FROZEN_AUDIO_AND_SENSOR_RUNTIME",
                "parameter_count_sum_not_resident_peak": audio_parameters + sensor_parameters,
                "torch_version": torch.__version__,
                "torchaudio_version": torchaudio.__version__,
                "numpy_version": np.__version__,
                "h5py_version": h5py.__version__,
                "psutil_version": psutil_version(),
                "transformers_version": __import__("transformers").__version__,
                "huggingface_hub_version": __import__("huggingface_hub").__version__,
                "python_version": platform.python_version(),
                "os": platform.platform(),
                "cpu": platform.processor(),
                "cuda_initialized": torch.cuda.is_initialized(),
                "peak_working_set_bytes": peak_working_set_bytes(),
                "member_payload_sha256": sha256_bytes(member_payload),
            }
        )
        if torch.cuda.is_initialized():
            raise ValueError("CUDA must remain uninitialized in this CPU-only smoke")
    except Exception as error:
        receipt.update({"status": "FAILED", "error": repr(error)})
        raise
    finally:
        receipt["ended_utc"] = datetime.now(UTC).isoformat()
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["script_sha256"] = sha256_file(Path(__file__))
        (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
