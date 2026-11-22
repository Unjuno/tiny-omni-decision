"""One-row raw-input runtime smoke using the first frozen TRAIN example only."""

# The direct-script entrypoint inserts the repository paths before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pretrained_reuse.package_spoken_palette_runtime import (
    CORPUS_ROOT,
    DEFAULT_OUTPUT,
    EXPECTED_CORPUS,
    sha256_file,
)  # noqa: E402
from tiny_omni_decision.spoken_palette_runtime import (
    SpokenPaletteRuntime,
    waveform_from_wav_bytes,
)  # noqa: E402

DEFAULT_AUDIO_ARCHIVE = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\lombard-grid-archives-01\audio.zip"
)
DEFAULT_AUDIO_LAYOUT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\lombard-grid-layout-02-media-gated\train.json"
)
DEFAULT_AUDIO_FEATURES = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\lombard-grid-whisper-features-01"
)
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT / "runtime-smoke-train-first.json"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_first_train_row(corpus_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = corpus_root / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED_CORPUS:
        raise ValueError("frozen task-corpus manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("evaluation_policy", {}).get("sealed_audit_opened") is not False
        or manifest.get("source", {}).get("original_calibration_validation_or_audit_labels_loaded")
        is not False
        or manifest.get("source", {}).get("source_split_used") != "original TRAIN only"
    ):
        raise ValueError("frozen corpus is not marked train-only and sealed-audit-safe")
    train_path = corpus_root / "train.jsonl"
    if sha256_file(train_path) != manifest["split_row_sha256"]["train"]:
        raise ValueError("frozen TRAIN JSONL hash mismatch")
    with train_path.open(encoding="utf-8") as stream:
        first_line = stream.readline()
    row = json.loads(first_line)
    if row.get("split") != "train" or row.get("audio_feature_index") != 0:
        raise ValueError("smoke row is not the first expected TRAIN/audio feature row")
    return row, manifest


def run(
    bundle_root: Path,
    *,
    corpus_root: Path = CORPUS_ROOT,
    audio_archive: Path = DEFAULT_AUDIO_ARCHIVE,
    audio_layout: Path = DEFAULT_AUDIO_LAYOUT,
    audio_features: Path = DEFAULT_AUDIO_FEATURES,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> dict[str, Any]:
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite runtime smoke result: {output_file}")
    row, manifest = _read_first_train_row(corpus_root)
    screenshot_path = (corpus_root / row["screenshot_path"]).resolve(strict=True)
    if corpus_root.resolve() not in screenshot_path.parents:
        raise ValueError("TRAIN screenshot path escaped the frozen corpus")
    screenshot_sha = sha256_file(screenshot_path)
    if screenshot_sha != row["screenshot_sha256"]:
        raise ValueError("TRAIN screenshot hash mismatch")

    train_media = json.loads(audio_layout.read_text(encoding="utf-8"))
    if not train_media or train_media[0].get("split") != "train":
        raise ValueError("audio layout is not the original TRAIN-only list")
    media = train_media[0]
    if media.get("source_id") != row["source_id"] or media.get("source_id") != "s11_l_bbab2n":
        raise ValueError("frozen first TRAIN audiovisual source identity mismatch")
    member = media["assets"]["audio"]["name"]
    with zipfile.ZipFile(audio_archive) as archive:
        info = archive.getinfo(member)
        if info.file_size != media["assets"]["audio"]["bytes"]:
            raise ValueError("TRAIN WAV member length differs from source receipt")
        if info.CRC != media["assets"]["audio"]["crc32"]:
            raise ValueError("TRAIN WAV member CRC differs from source receipt")
        wav_bytes = archive.read(member)
    wav_sha = _sha256_bytes(wav_bytes)
    waveform = waveform_from_wav_bytes(wav_bytes)

    # This indexed cache item is the same first TRAIN source; it is used only to
    # quantify CPU-FP32 versus the original frozen-feature extraction backend.
    pooled_path = audio_features / "pooled.npy"
    feature_result = json.loads((audio_features / "result.json").read_text(encoding="utf-8"))
    if sha256_file(pooled_path) != feature_result["feature_hashes"]["pooled.npy"]:
        raise ValueError("frozen audio feature-bank hash mismatch")
    bank = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
    if bank.shape[1] != 384 or bank.dtype != np.float32:
        raise ValueError("frozen audio feature-bank schema mismatch")
    reference = np.asarray(bank[row["audio_feature_index"]], dtype=np.float32)

    import torch

    started = time.perf_counter()
    runtime = SpokenPaletteRuntime(bundle_root)
    loaded_seconds = time.perf_counter() - started
    audio_start = time.perf_counter()
    audio_embedding = runtime._encode_audio(waveform)
    audio_seconds = time.perf_counter() - audio_start
    denominator = float(np.linalg.norm(audio_embedding) * np.linalg.norm(reference))
    cosine = (
        float(np.clip(np.dot(audio_embedding, reference) / denominator, -1.0, 1.0))
        if denominator
        else 0.0
    )
    options = [
        {
            "id": str(option["option_id"]),
            "text": f"Screen action candidate {index + 1}",
            "x_norm": float(option["x_norm"]),
            "y_norm": float(option["y_norm"]),
        }
        for index, option in enumerate(row["options"])
    ]
    inference_start = time.perf_counter()
    decision = json.loads(
        runtime.decide_waveform(
            waveform,
            screenshot_path,
            request_id="spoken-palette-runtime-train-smoke-0001",
            options=options,
        )
    )
    inference_seconds = time.perf_counter() - inference_start
    probabilities = [option["probability"] for option in decision["options"]]
    if (
        len(probabilities) != 4
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
        or decision["selected_option_id"] not in {option["id"] for option in options}
    ):
        raise ValueError("runtime smoke returned an invalid typed choice distribution")

    output = {
        "status": "PASS_RAW_INPUT_TRAIN_ONLY_SMOKE",
        "scope": "one mechanics smoke; no model selection and no quality evaluation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "corpus_manifest_sha256": sha256_file(corpus_root / "manifest.json"),
        "train_jsonl_sha256": manifest["split_row_sha256"]["train"],
        "sample_id": row["sample_id"],
        "source_id": row["source_id"],
        "split": "train",
        "audio_member": member,
        "audio_bytes_sha256": wav_sha,
        "screenshot_sha256": screenshot_sha,
        "audio_feature_index": int(row["audio_feature_index"]),
        "cpu_fp32_vs_training_audio_feature": {
            "max_absolute_error": float(np.max(np.abs(audio_embedding - reference))),
            "mean_absolute_error": float(np.mean(np.abs(audio_embedding - reference))),
            "cosine_similarity": cosine,
            "note": (
                "diagnostic only: runtime CPU FP32 versus training CUDA BF16; not bitwise parity"
            ),
        },
        "bundle_manifest_sha256": sha256_file(bundle_root / "runtime-bundle.json"),
        "runtime_load_seconds": loaded_seconds,
        "raw_input_inference_seconds": inference_seconds,
        "audio_encoder_seconds": audio_seconds,
        "decision_json": decision,
        "sealed_audit_opened": False,
        "evaluation_split_opened": False,
        "model_updated": False,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--audio-archive", type=Path, default=DEFAULT_AUDIO_ARCHIVE)
    parser.add_argument("--audio-layout", type=Path, default=DEFAULT_AUDIO_LAYOUT)
    parser.add_argument("--audio-features", type=Path, default=DEFAULT_AUDIO_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()
    result = run(
        args.bundle,
        corpus_root=args.corpus,
        audio_archive=args.audio_archive,
        audio_layout=args.audio_layout,
        audio_features=args.audio_features,
        output_file=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
