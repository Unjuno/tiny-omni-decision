"""Extract pinned frozen features for the bounded X-Capture train/val sample.

This is an input/runtime smoke only. It does not fit a model or calculate a
quality metric, and it refuses to touch any evaluation-partition media.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLE_ROOT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2-local/"
    "xcapture-sample-audit-20261128-attempt-04"
)
CROP_ROOT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2-local/"
    "xcapture-masked-crops-20261128-attempt-02"
)
TINYCLIP_ROOT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r1/tinyclip-39m19m-audit-01/model"
)
EFFICIENTAT_ROOT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1"
)
EXPECTED = {
    "sample_manifest": "17bacf80527a1aa0e1496eb330637d7f81b039ca68ffa28ccfabf9b33b240c0f",
    "crop_manifest": "6c37406b3e8011bab5fd564470b5b8d312d4016966906a1f2683f723d4453c84",
    "tinyclip_model": "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587",
    "efficientat_weights": "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e",
}
EFFICIENTAT_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_float_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read the dataset's little-endian IEEE-float WAV without a codec plugin."""
    payload = path.read_bytes()
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    cursor = 12
    format_fields = None
    audio_data = None
    while cursor + 8 <= len(payload):
        chunk_id = payload[cursor : cursor + 4]
        chunk_size = struct.unpack_from("<I", payload, cursor + 4)[0]
        start = cursor + 8
        end = start + chunk_size
        if end > len(payload):
            raise ValueError(f"truncated WAV chunk in {path}")
        chunk = payload[start:end]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                raise ValueError(f"truncated WAV format chunk in {path}")
            format_fields = struct.unpack_from("<HHIIHH", chunk)
        elif chunk_id == b"data":
            audio_data = chunk
        cursor = end + (chunk_size & 1)
    if format_fields is None or audio_data is None:
        raise ValueError(f"missing WAV format or data chunk: {path}")
    encoding, channels, sample_rate, _byte_rate, block_align, bits = format_fields
    if encoding != 3 or bits != 32 or channels < 1 or block_align != channels * 4:
        raise ValueError(f"expected IEEE-float32 WAV, got fmt={format_fields}: {path}")
    if len(audio_data) % block_align:
        raise ValueError(f"unaligned WAV payload: {path}")
    waveform = np.frombuffer(audio_data, dtype="<f4").reshape(-1, channels)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError(f"invalid float samples in {path}")
    mono = waveform.mean(axis=1, dtype=np.float64).astype(np.float32)
    return mono, sample_rate


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output}")

    sample_manifest_path = SAMPLE_ROOT / "sample-audit.json"
    crop_manifest_path = CROP_ROOT / "masked-crops.json"
    verify_hash(sample_manifest_path, EXPECTED["sample_manifest"])
    verify_hash(crop_manifest_path, EXPECTED["crop_manifest"])
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    if sample_manifest["dataset_revision"] != "b4468bf999aac250f34ef1b02459d902492a068f":
        raise ValueError("unexpected X-Capture revision")
    if set(crop_manifest["partitions"]) != {"train", "validation"}:
        raise ValueError("crop manifest includes a non-development partition")
    efficientat_source_commit = subprocess.check_output(
        ["git", "-C", str(EFFICIENTAT_ROOT / "source"), "rev-parse", "HEAD"], text=True
    ).strip()
    efficientat_source_dirty = subprocess.check_output(
        [
            "git", "-C", str(EFFICIENTAT_ROOT / "source"), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if efficientat_source_commit != EFFICIENTAT_SOURCE_COMMIT or efficientat_source_dirty:
        raise ValueError("EfficientAT source is not the pinned pristine revision")

    sample_rows = {
        (row["partition"], row["object_id"], row["point_id"]): row
        for row in sample_manifest["sample_points"]
    }
    crop_rows = {
        (row["partition"], row["object_id"], row["point_id"]): row
        for row in crop_manifest["rows"]
    }
    if len(sample_rows) != 20 or len(crop_rows) != 20 or sample_rows.keys() != crop_rows.keys():
        raise ValueError("expected the same 20 frozen train/validation points in both manifests")
    if {key[0] for key in sample_rows} != {"train", "validation"}:
        raise ValueError("unexpected split in sample manifest")
    if len({key[1] for key in sample_rows}) != 20:
        raise ValueError("sample must contain 20 distinct object identities")

    sys.path.insert(0, str(Path("C:/CodexArtifacts/pretrained-reuse-r1/python-deps")))
    sys.path.insert(0, str(EFFICIENTAT_ROOT / "source"))
    import torch
    import torchaudio
    from models.dymn.model import get_model
    from models.preprocess import AugmentMelSTFT
    from transformers import CLIPImageProcessor, CLIPModel

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tinyclip_weights = TINYCLIP_ROOT / "model.safetensors"
    efficientat_weights = EFFICIENTAT_ROOT / "dymn04_as.pt"
    verify_hash(tinyclip_weights, EXPECTED["tinyclip_model"])
    verify_hash(efficientat_weights, EXPECTED["efficientat_weights"])

    model = (
        CLIPModel.from_pretrained(TINYCLIP_ROOT, local_files_only=True)
        .eval()
        .requires_grad_(False)
    )
    image_processor = CLIPImageProcessor.from_pretrained(TINYCLIP_ROOT, local_files_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        audio_model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
    audio_model.load_state_dict(
        torch.load(efficientat_weights, map_location="cpu", weights_only=True), strict=True
    )
    audio_model.eval().requires_grad_(False)
    mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False)
    tinyclip_parameters = sum(parameter.numel() for parameter in model.parameters())
    tinyclip_vision_parameters = sum(
        parameter.numel() for parameter in model.vision_model.parameters()
    )
    efficientat_parameters = sum(parameter.numel() for parameter in audio_model.parameters())
    if tinyclip_parameters != 83_101_185 or efficientat_parameters != 1_961_239:
        raise ValueError("loaded checkpoint architecture parameter count mismatch")

    image_vectors: list[np.ndarray] = []
    tactile_vectors: list[np.ndarray] = []
    audio_vectors: list[np.ndarray] = []
    records = []
    started = time.perf_counter()
    audio_seconds = 0.0
    image_seconds = 0.0
    tactile_seconds = 0.0
    image_sample_ids = {
        (row["partition"], row["object_id"], row["point_id"]): row["sample_id"]
        for row in sample_manifest["files"]
        if row["modality"] == "image"
    }

    for key in sorted(sample_rows, key=lambda item: image_sample_ids[item]):
        point = sample_rows[key]
        crop = crop_rows[key]
        sid = image_sample_ids[key]
        image_path = CROP_ROOT / crop["output_path"]
        if sha256_file(image_path) != crop["output_sha256"]:
            raise ValueError(f"masked-crop hash mismatch for sample {sid}")
        audio_row = next(
            row for row in sample_manifest["files"]
            if row["sample_id"] == sid and row["modality"] == "audio"
        )
        tactile_row = next(
            row for row in sample_manifest["files"]
            if row["sample_id"] == sid and row["modality"] == "tactile"
        )
        audio_path = SAMPLE_ROOT / audio_row["output_path"]
        tactile_path = SAMPLE_ROOT / tactile_row["output_path"]
        verify_hash(audio_path, audio_row["sha256"])
        verify_hash(tactile_path, tactile_row["sha256"])

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        tick = time.perf_counter()
        with torch.inference_mode():
            image_feature = model.vision_model(**inputs).pooler_output[0]
        image_seconds += time.perf_counter() - tick

        with Image.open(tactile_path) as source:
            tactile = source.convert("RGB")
        tactile_inputs = image_processor(images=tactile, return_tensors="pt")
        tick = time.perf_counter()
        with torch.inference_mode():
            tactile_feature = model.vision_model(**tactile_inputs).pooler_output[0]
        tactile_seconds += time.perf_counter() - tick

        waveform_array, sample_rate = read_float_wav(audio_path)
        waveform = torch.from_numpy(waveform_array).unsqueeze(0)
        source_sample_rate = sample_rate
        if sample_rate != 32000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 32000)
            sample_rate = 32000
        tick = time.perf_counter()
        with torch.inference_mode():
            spectrogram = mel(waveform)
            _, audio_feature = audio_model(spectrogram.unsqueeze(0))
        audio_seconds += time.perf_counter() - tick

        features = {
            "image": image_feature.detach().cpu().numpy().astype(np.float32),
            "tactile": tactile_feature.detach().cpu().numpy().astype(np.float32),
            "audio": audio_feature.detach().cpu().numpy().reshape(-1).astype(np.float32),
        }
        for modality, vector in features.items():
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"invalid {modality} feature for sample {sid}")
        image_vectors.append(features["image"])
        tactile_vectors.append(features["tactile"])
        audio_vectors.append(features["audio"])
        records.append(
            {
                **point,
                "input_sha256": {
                    "masked_image": crop["output_sha256"],
                    "audio": audio_row["sha256"],
                    "tactile": tactile_row["sha256"],
                },
                "source_sample_rate_hz": source_sample_rate,
                "encoder_sample_rate_hz": sample_rate,
                "audio_duration_seconds": len(waveform_array) / source_sample_rate,
                "feature_sha256": {
                    key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in features.items()
                },
            }
        )

    elapsed = time.perf_counter() - started
    output.mkdir(parents=True)
    arrays = {
        "image": np.stack(image_vectors),
        "audio": np.stack(audio_vectors),
        "tactile": np.stack(tactile_vectors),
    }
    array_hashes = {}
    for modality, values in arrays.items():
        path = output / f"{modality}-features.npy"
        np.save(path, values, allow_pickle=False)
        restored = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(restored, values)
        array_hashes[path.name] = sha256_file(path)

    report = {
        "status": "PASS_FROZEN_FEATURE_RUNTIME_SMOKE_ONLY",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_revision": sample_manifest["dataset_revision"],
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "crop_manifest_sha256": sha256_file(crop_manifest_path),
        "tinyclip_revision": "07a4b0bc751cb64fecd2b661c048c1dd98d69444",
        "tinyclip_model_sha256": sha256_file(tinyclip_weights),
        "efficientat_source_revision": EFFICIENTAT_SOURCE_COMMIT,
        "efficientat_source_dirty": False,
        "efficientat_weights_sha256": sha256_file(efficientat_weights),
        "device": "cpu",
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": __import__("transformers").__version__,
        "sample_count": len(records),
        "loaded_parameter_counts": {
            "tinyclip_full": tinyclip_parameters,
            "tinyclip_vision_tower": tinyclip_vision_parameters,
            "efficientat_dymn04": efficientat_parameters,
        },
        "partition_counts": {
            partition: sum(row["partition"] == partition for row in records)
            for partition in ("train", "validation")
        },
        "unique_object_count": len({row["object_id"] for row in records}),
        "feature_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "feature_array_sha256": array_hashes,
        "elapsed_seconds": elapsed,
        "forward_seconds": {
            "image": image_seconds,
            "audio": audio_seconds,
            "tactile": tactile_seconds,
        },
        "records": records,
        "quality_evaluation": False,
        "training": False,
        "evaluation_partition_accessed": False,
        "limitations": [
            "Only 20 preselected train/validation objects were processed.",
            "Image and tactile use TinyCLIP's frozen visual tower, not a tactile-pretrained model.",
            "Features establish runtime compatibility, not task usefulness or retrieval quality.",
        ],
        "cost_usd": 0,
    }
    report_path = output / "feature-smoke.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "C:/CodexArtifacts/pretrained-reuse-r2-local/"
            "xcapture-frozen-feature-smoke-20261128-attempt-01"
        ),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
