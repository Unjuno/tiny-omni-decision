"""Run a fixed train-row ADL audio input through a pinned frozen CPU encoder."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_adl_participant_split import participant_split
from scripts.audit_adl_sensor_identity import ARCHIVE_SHA
from scripts.extract_perception_temporal_match_efficientat import (
    SOURCE_COMMIT,
    WEIGHTS_SHA256,
    sha256_file,
    state_sha,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1")
ARCHIVE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-archive-20261027-attempt-01/23_take-from-stove.zip"
)
SOURCE = ROOT / "efficientat-audit-v1/source"
WEIGHTS = ROOT / "efficientat-audit-v1/dymn04_as.pt"
PROTOCOL = Path("docs/pretrained_reuse/R2_ADL_AUDIO_RUNTIME_PROTOCOL_20261027.md")
PROTOCOL_SHA = "573009755c681b153340277ca03ccc03db310b9d60d903cafa7d3d42293cafcf"


def first_train_row(participants):
    return next(
        i for i, participant in enumerate(participants) if participant_split(participant) == "train"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("output already exists")
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_sha != PROTOCOL_SHA:
        raise ValueError("preregistered protocol hash mismatch")
    if sha256_file(ARCHIVE) != ARCHIVE_SHA or sha256_file(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError("archive or checkpoint hash mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    ).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("source revision mismatch or dirty checkout")
    output.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {
        "status": "RUNNING",
        "archive_sha256": ARCHIVE_SHA,
        "checkpoint_sha256": WEIGHTS_SHA256,
        "source_commit": commit,
        "protocol_sha256_lf": protocol_sha,
        "script_sha256": sha256_file(Path(__file__)),
        "training": False,
        "validation_waveform_read": False,
        "sealed_audit_opened": False,
    }
    try:
        sys.path.append(str(ROOT / "videoprism-base-r1/venv/Lib/site-packages"))
        import h5py

        with zipfile.ZipFile(ARCHIVE) as archive:
            members = sorted(i for i in archive.namelist() if i.endswith(".hdf5"))
            if len(members) != 5:
                raise ValueError("expected five HDF5 members")
            payload = archive.read(members[0])
        with h5py.File(io.BytesIO(payload), "r") as h:
            row = first_train_row(h["participant"][:])
            audio = np.asarray(h["audio"][row])
            axes = [np.asarray(h[f"vibration-{axis}"][row]) for axis in "xyz"]
            rate = float(h["vibration_sample_rate"][row])
        if audio.dtype != np.int16 or audio.ndim != 1 or len(audio) < 32000:
            raise ValueError("audio must contain at least two seconds of int16 samples")
        if not np.isfinite(rate) or rate <= 0 or any(a.ndim != 1 or len(a) == 0 for a in axes):
            raise ValueError("invalid vibration metadata")
        sys.path.insert(0, str(ROOT / "python-deps"))
        sys.path.insert(0, str(SOURCE))
        import torch
        import torchaudio

        with contextlib.redirect_stdout(io.StringIO()):
            from models.dymn.model import get_model
            from models.preprocess import AugmentMelSTFT

            model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        torch.set_num_threads(4)
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
        mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False)
        before = state_sha(model)
        with torch.inference_mode():
            samples = torch.from_numpy(audio[:32000].astype(np.float32) / 32768).unsqueeze(0)
            samples = torchaudio.functional.resample(samples, 16000, 32000)
            _, embedding = model(mel(samples).unsqueeze(1))
        features = embedding.float().numpy()
        if (
            features.shape != (1, 384)
            or not np.isfinite(features).all()
            or not np.linalg.norm(features) > 0
        ):
            raise ValueError("invalid features")
        after = state_sha(model)
        if before != after or any(p.requires_grad for p in model.parameters()):
            raise ValueError("frozen state changed")
        feature_path = output / "features.npy"
        with feature_path.open("xb") as f:
            np.save(f, features, allow_pickle=False)
        receipt.update(
            {
                "status": "PASS_ONE_TRAIN_ROW_FROZEN_AUDIO_RUNTIME",
                "row_index": row,
                "member": members[0],
                "member_sha256": hashlib.sha256(payload).hexdigest(),
                "audio_samples": len(audio),
                "source_audio_rate_hz_declared": 16000,
                "audio_duration_seconds_declared_rate": len(audio) / 16000,
                "vibration_axis_lengths": [len(a) for a in axes],
                "vibration_rate": rate,
                "vibration_axis_durations_seconds": [len(a) / rate for a in axes],
                "audio_window_seconds": 2,
                "feature_shape": list(features.shape),
                "feature_norm": float(np.linalg.norm(features)),
                "feature_sha256": sha256_file(feature_path),
                "model_state_before": before,
                "model_state_after": after,
                "parameters": sum(p.numel() for p in model.parameters()),
                "cuda_initialized": torch.cuda.is_initialized(),
                "torch_version": torch.__version__,
                "h5py_version": h5py.__version__,
                "numpy_version": np.__version__,
            }
        )
    except Exception as error:
        receipt.update({"status": "FAILED", "error": repr(error)})
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
