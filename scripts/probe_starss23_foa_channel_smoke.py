"""One-window CPU smoke for all four frozen STARSS23 FOA channel features."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_features import load_encoder
from scripts.extract_starss23_foa_channel_features import (
    AUDIO_SHA,
    AUDIO_SOURCE,
    FOA_CHANNEL_ORDER,
    encode_foa_window,
)
from scripts.fetch_starss23_frozen_media import write_json

WINDOW_RESULT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-window-smoke-01/result.json"
)
WINDOW_ID = "fold3_room12_mix001:t0000-0005"
WINDOW_PATH = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-window-smoke-01/fold3_room12_mix001-0000.npz"
)
WINDOW_SHA256 = "05f4be05373f280c4cbff0e0cb1d4ed9d50d33752cf97b7ff04376954e616c55"
CHECKPOINT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt"
)
SOURCE_ROOT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/source"
)
FEATURE_RESULT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-audio-corpus-features-v1/result.json"
)
FEATURE_ARRAY = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-audio-corpus-features-v1/features.npy"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if sha256_file(WINDOW_PATH) != WINDOW_SHA256:
        raise ValueError("fixed dev-train window hash mismatch")
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AUDIO_SOURCE or sha256_file(CHECKPOINT) != AUDIO_SHA:
        raise ValueError("pinned DyMN source/checkpoint mismatch")
    window_result = json.loads(WINDOW_RESULT.read_text(encoding="utf-8"))
    selected = [row for row in window_result["windows"] if row.get("id") == WINDOW_ID]
    if len(selected) != 1 or selected[0].get("split") != "dev-train":
        raise ValueError("fixed audio smoke is not a dev-train window")
    feature_report = json.loads(FEATURE_RESULT.read_text(encoding="utf-8"))
    if (
        feature_report.get("status") != "PASS"
        or feature_report.get("modality") != "audio"
        or sha256_file(FEATURE_ARRAY) != feature_report.get("feature_sha256")
    ):
        raise ValueError("existing audio feature reference is invalid")
    feature_rows = [row for row in feature_report["features"] if row.get("id") == WINDOW_ID]
    if len(feature_rows) != 1 or feature_rows[0].get("split") != "train":
        raise ValueError("missing fixed channel-0 feature reference")
    reference_matrix = np.load(FEATURE_ARRAY, mmap_mode="r", allow_pickle=False)
    reference_index = feature_report["features"].index(feature_rows[0])
    started = time.perf_counter()
    report = {
        "status": "RUNNING",
        "scope": "one fixed dev-train FOA window; CPU only; no annotations or fitting",
        "window_id": WINDOW_ID,
        "window_sha256": WINDOW_SHA256,
        "window_result_sha256": sha256_file(WINDOW_RESULT),
        "checkpoint_sha256": AUDIO_SHA,
        "source_commit": AUDIO_SOURCE,
        "channel_order": list(FOA_CHANNEL_ORDER),
        "reference_result_sha256": sha256_file(FEATURE_RESULT),
        "reference_features_sha256": sha256_file(FEATURE_ARRAY),
        "runner_sha256": sha256_file(Path(__file__)),
        "device": "cpu",
        "training": False,
        "annotations_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "progress.json", report)
    try:
        sys.path.insert(0, str(SOURCE_ROOT))
        with np.load(WINDOW_PATH, allow_pickle=False) as arrays:
            audio = np.asarray(arrays["audio"], dtype=np.float32)
        encode, shape = load_encoder(
            argparse.Namespace(modality="audio", source_root=SOURCE_ROOT, checkpoint=CHECKPOINT),
            report,
        )
        if shape != (384,):
            raise ValueError("unexpected pinned DyMN feature shape")
        features = encode_foa_window(audio, encode)
        reference = np.asarray(reference_matrix[reference_index], dtype=np.float32)
        delta = float(np.max(np.abs(features[0] - reference)))
        if not np.allclose(features[0], reference, rtol=1e-5, atol=1e-6):
            raise ValueError("first-channel feature differs from existing audio cache")
        path = args.output_dir / "foa-channel-features.npy"
        np.save(path, features, allow_pickle=False)
        if not np.array_equal(features, np.load(path, allow_pickle=False)):
            raise ValueError("FOA feature save/reload mismatch")
        report.update(
            status="PASS",
            feature_shape=list(features.shape),
            feature_sha256=sha256_file(path),
            channel0_reference_max_abs_diff=delta,
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        if args.output_dir.exists():
            write_json(args.output_dir / "result.json", report)
            write_json(args.output_dir / "progress.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
