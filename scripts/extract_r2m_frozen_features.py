"""Cache frozen EfficientAT audio and deterministic vibration tokens by activity."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio

from tiny_omni_decision.r2m_pairs import sample_id

SAMPLE_RATE_HZ = 16000
TARGET_AUDIO_RATE_HZ = 32000
SENSOR_TOKENS = 128
ENCODER_PARAMETERS = 1_961_239
EXPECTED_CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
PARTICIPANT_SPLIT_SHA256 = "0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resample_sensor(values: np.ndarray, tokens: int = SENSOR_TOKENS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        raise ValueError("sensor values must be [time>=2, xyz]")
    if not np.isfinite(values).all():
        raise ValueError("sensor values must be finite")
    source_times = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    target_times = np.linspace(0.0, 1.0, tokens, dtype=np.float64)
    return np.stack(
        [np.interp(target_times, source_times, values[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)


def build_audio_encoder(checkpoint: Path, source: Path):
    import sys

    torch.set_num_threads(1)
    sys.path.insert(0, str(source))
    from models.dymn.model import get_model
    from models.preprocess import AugmentMelSTFT

    with contextlib.redirect_stdout(io.StringIO()):
        model = get_model(width_mult=0.4, pretrained_name=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    mel = AugmentMelSTFT(sr=TARGET_AUDIO_RATE_HZ)
    mel.eval()
    resample = torchaudio.transforms.Resample(SAMPLE_RATE_HZ, TARGET_AUDIO_RATE_HZ)
    return model, mel, resample


def encode_audio(waveform: np.ndarray, model, mel, resample) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.int16)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("audio waveform must be a nonempty int16 vector")
    signal = torch.from_numpy(values.astype(np.float32) / 32768.0).unsqueeze(0)
    with torch.inference_mode():
        signal = resample(signal)
        spec = mel(signal)
        feature_map = model._feature_forward(spec.unsqueeze(0))
        features = feature_map.mean(dim=2).squeeze(0).transpose(0, 1).contiguous()
    result = features.cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[1] != 384 or not np.isfinite(result).all():
        raise ValueError("EfficientAT produced an invalid temporal feature tensor")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--efficientat-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit("refusing to overwrite; use a new output path or --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    metadata_path = args.pair_index_dir / "allowed-trial-metadata.json"
    pair_path = args.pair_index_dir / "pair-index.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(Path(
        "manifests/pretrained_reuse/r2m-audio-vibration-split-v2-20261120.json"
    ).read_text(encoding="utf-8"))
    declared_split_hash = split_manifest.get("manifest_sha256")
    split_payload = {key: value for key, value in split_manifest.items()
                     if key != "manifest_sha256"}
    split_encoded = json.dumps(
        split_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if (declared_split_hash != PARTICIPANT_SPLIT_SHA256
            or hashlib.sha256(split_encoded).hexdigest() != declared_split_hash):
        raise SystemExit("participant split differs from frozen v2")
    participant_to_split = {
        int(participant): split
        for split, participants in split_manifest["splits"].items()
        for participant in participants
    }
    for row in metadata:
        participant = int(row["participant_id"])
        if (participant_to_split.get(participant) != row["split"]
                or row["split"] not in {"train", "calibration", "validation"}):
            raise SystemExit("allowed-trial metadata violates frozen participant split")
    if pairs.get("sealed_audit_accessed") is not False:
        raise SystemExit("pair index does not establish sealed-audit isolation")
    declared_pair_hash = pairs.get("manifest_sha256")
    pair_payload = {key: value for key, value in pairs.items()
                    if key != "manifest_sha256"}
    canonical_pair_payload = json.dumps(
        pair_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if hashlib.sha256(canonical_pair_payload).hexdigest() != declared_pair_hash:
        raise SystemExit("pair-index internal hash verification failed")
    if metadata_path.parent != pair_path.parent:
        raise SystemExit("pair-index artifacts must come from one frozen directory")
    trial_by_key = {
        (str(row["activity"]), int(row["participant_id"]), int(row["trial_id"])): {
            **row,
            "sample_id": sample_id(row),
        }
        for row in metadata
    }
    if len(trial_by_key) != len(metadata):
        raise SystemExit("duplicate trial key in allowed metadata")
    rows_by_archive: dict[str, dict[str, dict[str, object]]] = {}
    for pair in pairs.get("pairs", []):
        ids = (pair["anchor_sample_id"], pair["positive_sensor_sample_id"],
               pair["negative_sensor_sample_id"])
        for identifier in ids:
            activity, participant_part, trial_part = str(identifier).split("|")
            participant_id = int(participant_part.removeprefix("participant="))
            trial_id = int(trial_part.removeprefix("trial="))
            key = (activity, participant_id, trial_id)
            row = trial_by_key.get(key)
            if row is None or row["split"] != pair["split"]:
                raise SystemExit("pair references a missing or cross-split source trial")
            rows_by_archive.setdefault(str(row["source_archive"]), {})[
                str(row["sample_id"])
            ] = row

    receipt = json.loads(
        Path("manifests/pretrained_reuse/r2m-audio-vibration-payload-verification-20261120.json")
        .read_text(encoding="utf-8")
    )
    expected_archives = {item["file"]: item for item in receipt["files"]}
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("EfficientAT checkpoint differs from the audited revision")
    model, mel, resample = build_audio_encoder(args.checkpoint, args.efficientat_source)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != ENCODER_PARAMETERS or any(p.requires_grad for p in model.parameters()):
        raise SystemExit("frozen audio encoder parameter gate failed")

    index: list[dict[str, object]] = []
    started = time.perf_counter()
    for archive_name, targets in sorted(rows_by_archive.items()):
        archive = args.artifact_root / "source" / archive_name
        expected = expected_archives.get(archive_name)
        if not expected or sha256_file(archive) != expected["sha256"]:
            raise SystemExit(f"archive digest gate failed: {archive_name}")
        feature_path = features_dir / f"{archive_name[:-4]}.npz"
        sidecar_path = features_dir / f"{archive_name[:-4]}.json"
        if feature_path.exists() or sidecar_path.exists():
            if not args.resume or not feature_path.exists() or not sidecar_path.exists():
                raise SystemExit(f"partial/existing activity output: {archive_name}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (sidecar.get("archive_sha256") != expected["sha256"]
                    or sidecar.get("checkpoint_sha256") != checkpoint_hash
                    or sidecar.get("pair_index_sha256") != pairs["manifest_sha256"]
                    or sidecar.get("feature_sha256") != sha256_file(feature_path)):
                raise SystemExit(f"resume artifact integrity gate failed: {archive_name}")
            index.append(sidecar)
            continue

        with zipfile.ZipFile(archive) as zf:
            members = [path for path in zf.namelist()
                       if path.endswith("_multisensor-1.hdf5")]
            if len(members) != 1:
                raise SystemExit(f"expected one multisensor-1 member: {archive_name}")
            raw_hdf5 = zf.read(members[0])
        encoded: list[np.ndarray] = []
        sensors: list[np.ndarray] = []
        ids: list[str] = []
        pids: list[int] = []
        trials: list[int] = []
        splits: list[str] = []
        activities: list[str] = []
        durations: list[float] = []
        with h5py.File(io.BytesIO(raw_hdf5), "r") as data:
            participant_ids = data["participant"][:]
            trial_ids = data["trial"][:]
            for row_index, (raw_pid, raw_trial) in enumerate(
                zip(participant_ids, trial_ids, strict=True)
            ):
                key = (archive_name[:-4], int(raw_pid), int(raw_trial))
                candidate = targets.get(
                    f"{key[0]}|participant={key[1]}|trial={key[2]}"
                )
                if candidate is None:
                    continue
                # The ID/split allowlist is checked before signal arrays are read.
                audio = np.asarray(data["audio"][row_index], dtype=np.int16)
                vibration = [
                    np.asarray(data[f"vibration-{axis}"][row_index], dtype=np.float32)
                    for axis in "xyz"
                ]
                if (audio.size == 0 or any(axis.size < 2 for axis in vibration)
                        or len({axis.size for axis in vibration}) != 1):
                    raise SystemExit(f"invalid signal shapes for {key}")
                rate = float(data["vibration_sample_rate"][row_index])
                duration = audio.size / SAMPLE_RATE_HZ
                sensor_duration = vibration[0].size / rate
                if (not math.isfinite(rate) or rate <= 0 or sensor_duration <= 0
                        or abs(duration / sensor_duration - 1.0) > 0.005):
                    raise SystemExit(f"audio/vibration timing gate failed for {key}")
                audio_features = encode_audio(audio, model, mel, resample)
                sensor_features = resample_sensor(np.stack(vibration, axis=1))
                identifier = str(candidate["sample_id"])
                if not np.isfinite(sensor_features).all():
                    raise SystemExit(f"non-finite sensor tokens for {identifier}")
                encoded.append(audio_features)
                sensors.append(sensor_features)
                ids.append(identifier)
                pids.append(key[1])
                trials.append(key[2])
                splits.append(str(candidate["split"]))
                activities.append(key[0])
                durations.append(duration)
        if set(ids) != set(targets) or len(ids) != len(targets):
            raise SystemExit(f"source rows missing or duplicated for {archive_name}")

        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([item.shape[0] for item in encoded])
        flat_audio = np.concatenate(encoded, axis=0).astype(np.float32, copy=False)
        flat_sensor = np.stack(sensors).astype(np.float32, copy=False)
        temporary = feature_path.with_suffix(".partial.npz")
        np.savez_compressed(
            temporary,
            sample_ids=np.asarray(ids, dtype="U128"),
            participant_ids=np.asarray(pids, dtype=np.int16),
            trial_ids=np.asarray(trials, dtype=np.int32),
            splits=np.asarray(splits, dtype="U16"),
            activities=np.asarray(activities, dtype="U64"),
            durations_seconds=np.asarray(durations, dtype=np.float32),
            audio_values=flat_audio,
            audio_offsets=offsets,
            sensor_values=flat_sensor,
        )
        os.replace(temporary, feature_path)
        sidecar = {
            "schema": "r2m_frozen_features_by_activity_v1",
            "activity_archive": archive_name,
            "archive_sha256": expected["sha256"],
            "pair_index_sha256": pairs["manifest_sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "encoder_parameters": parameters,
            "encoder_trainable_parameters": 0,
            "precision": "CPU FP32",
            "samples": len(ids),
            "audio_feature_shape": [int(flat_audio.shape[0]), 384],
            "sensor_feature_shape": list(flat_sensor.shape),
            "raw_waveforms_persisted": False,
            "sealed_audit_accessed": False,
            "feature_sha256": sha256_file(feature_path),
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        index.append(sidecar)
        elapsed = time.perf_counter() - started
        done = sum(int(item["samples"]) for item in index)
        print(
            f"cached {archive_name}: {len(ids)} samples; "
            f"{done} total; {elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "schema": "r2m_frozen_feature_cache_v1",
        "pair_index_sha256": pairs["manifest_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "encoder_parameters": parameters,
        "encoder_trainable_parameters": 0,
        "sample_count_by_split": {
            "train": 0,
            "calibration": 0,
            "validation": 0,
        },
        "archives": index,
        "sealed_audit_accessed": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    # Compute split counts from the immutable activity sidecars.
    counts = {split: 0 for split in ("train", "calibration", "validation")}
    for sidecar in index:
        with np.load(features_dir / f"{sidecar['activity_archive'][:-4]}.npz") as cached:
            for split in counts:
                counts[split] += int(np.count_nonzero(cached["splits"] == split))
    summary["sample_count_by_split"] = counts
    (args.output_dir / "feature-cache-index.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sample_count_by_split": counts,
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "sealed_audit_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
