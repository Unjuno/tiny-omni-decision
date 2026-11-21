"""Reproduce the pinned train-only EfficientAT feature smoke without saving tensors."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from extract_r2m_frozen_features import build_audio_encoder, encode_audio, resample_sensor

PREVIOUS_SMOKE_FEATURE_SHA256 = (
    "34528184e6cac158fdfa69b7beaa5ca03d7a1c3fd830a4417da848c5e4e08695"
)
CURRENT_SCRIPT_REPEAT_SHA256 = (
    "e662c72ef734c18ca22ee65cc0431743786e33651ea22e788b1ae9591f64349b"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--efficientat-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.artifact_root / "source" / "01_walk.zip"
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist()
                   if name.endswith("_multisensor-1.hdf5")]
        if len(members) != 1:
            raise SystemExit("expected one multisensor-1 member")
        payload = zf.read(members[0])
    audio = None
    vibration = None
    with h5py.File(io.BytesIO(payload), "r") as data:
        participant_ids = data["participant"][:]
        trial_ids = data["trial"][:]
        for row_index, (participant_id, trial_id) in enumerate(
            zip(participant_ids, trial_ids, strict=True)
        ):
            if int(participant_id) == 2 and int(trial_id) == 1649:
                audio = np.asarray(data["audio"][row_index], dtype=np.int16)
                vibration = np.stack([
                    np.asarray(data[f"vibration-{axis}"][row_index], dtype=np.float32)
                    for axis in "xyz"
                ], axis=1)
                break
    if audio is None or vibration is None:
        raise SystemExit("frozen permitted train sample was not found")

    model, mel, resample = build_audio_encoder(args.checkpoint, args.efficientat_source)
    features = encode_audio(audio, model, mel, resample)
    digest = hashlib.sha256(features.tobytes(order="C")).hexdigest()
    sensor = resample_sensor(vibration)
    result = {
        "schema": "r2m_frozen_feature_extractor_smoke_v1",
        "status": "PASS" if digest == CURRENT_SCRIPT_REPEAT_SHA256 else "MISMATCH",
        "sample_id": "01_walk|participant=2|trial=1649",
        "split": "train",
        "audio_feature_shape": list(features.shape),
        "audio_feature_sha256": digest,
        "current_script_repeat_expected_sha256": CURRENT_SCRIPT_REPEAT_SHA256,
        "previous_smoke_sha256": PREVIOUS_SMOKE_FEATURE_SHA256,
        "previous_smoke_hash_matches": digest == PREVIOUS_SMOKE_FEATURE_SHA256,
        "sensor_feature_shape": list(sensor.shape),
        "sensor_finite": bool(np.isfinite(sensor).all()),
        "raw_signals_persisted": False,
        "device": "CPU",
        "precision": "FP32",
        "torch_num_threads": torch.get_num_threads(),
        "sealed_audit_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
