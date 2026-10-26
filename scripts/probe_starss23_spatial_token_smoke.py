"""One-window CPU smoke for preserving VideoPrism horizontal patch bands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from scripts.extract_frozen_videoprism_features import CHECKPOINT_SHA256, SOURCE_COMMIT, sha256_file
from tiny_omni_decision.starss23_spatial_features import panorama_sector_features

WINDOWS_RESULT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-window-smoke-01/result.json"
)
WINDOW_ID = "fold3_room12_mix001:t0000-0005"
WINDOW_NPZ = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-window-smoke-01/fold3_room12_mix001-0000.npz"
)
FEATURES_RESULT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-video-corpus-features-v1/result.json"
)
FEATURES_ARRAY = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-video-corpus-features-v1/features.npy"
)
SOURCE_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
CHECKPOINT = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz")


def _validate_training_inputs() -> tuple[dict, dict]:
    windows = json.loads(WINDOWS_RESULT.read_text(encoding="utf-8"))
    if windows.get("status") != "PASS":
        raise ValueError("STARSS23 window smoke is not complete")
    rows = [row for row in windows["windows"] if row.get("id") == WINDOW_ID]
    if len(rows) != 1 or rows[0].get("split") != "dev-train":
        raise ValueError("expected exactly one pinned dev-train window")
    window = rows[0]

    features = json.loads(FEATURES_RESULT.read_text(encoding="utf-8"))
    if features.get("status") != "PASS" or features.get("modality") != "video":
        raise ValueError("reference pooled video features are not complete")
    entries = [row for row in features["features"] if row.get("id") == WINDOW_ID]
    if len(entries) != 1 or entries[0].get("split") != "train":
        raise ValueError("expected exactly one train feature reference")
    return window, entries[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    if (
        subprocess.check_output(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True)
        .strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("VideoPrism source revision mismatch")
    window, pooled_entry = _validate_training_inputs()
    window_sha = sha256_file(WINDOW_NPZ)
    started = time.perf_counter()

    sys.path.insert(0, str(SOURCE_ROOT))
    with np.load(WINDOW_NPZ, allow_pickle=False) as arrays:
        if "video" not in arrays.files:
            raise ValueError("training window lacks video frames")
        video = np.asarray(arrays["video"])
    if video.shape != (8, 960, 1920, 3) or video.dtype != np.uint8:
        raise ValueError(f"unexpected fixed-window video input {video.shape}/{video.dtype}")

    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("CPU-only smoke must not initialize a GPU device")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(CHECKPOINT)
    )
    forward = jax.jit(lambda x: model.apply(variables, x, train=False))
    inputs = np.asarray(tf.image.resize(video, (288, 288)), dtype=np.float32)[None] / 255.0
    result = forward(jnp.asarray(inputs))
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
    tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
    if tokens.shape != (1, 2048, 768) or not np.isfinite(tokens).all():
        raise ValueError(f"unexpected VideoPrism output {tokens.shape}")
    sectors = panorama_sector_features(tokens, frame_count=8)

    reference = np.load(FEATURES_ARRAY, mmap_mode="r", allow_pickle=False)
    if reference.shape[0] != len(json.loads(FEATURES_RESULT.read_text())["features"]):
        raise ValueError("pooled feature rows differ from the pinned feature index")
    pooled_reference = np.asarray(reference[0], dtype=np.float32)
    pooled_replay = tokens.mean(axis=1).squeeze(0).astype(np.float32)
    pooled_max_abs_diff = float(np.max(np.abs(pooled_reference - pooled_replay)))
    if not np.allclose(pooled_reference, pooled_replay, rtol=1e-5, atol=1e-6):
        raise ValueError("one-window mean-pooled replay differs from existing frozen feature")

    args.output.mkdir(parents=True, exist_ok=False)
    sector_path = args.output / "azimuth-sector-features.npy"
    np.save(sector_path, sectors, allow_pickle=False)
    report = {
        "status": "PASS",
        "scope": "one dev-train STARSS23 window; no labels read and no training",
        "window_id": WINDOW_ID,
        "window_npz_sha256": window_sha,
        "runner_sha256": sha256_file(Path(__file__)),
        "window_result_sha256": sha256_file(WINDOWS_RESULT),
        "recording": window["recording"],
        "room": window["room"],
        "split": window["split"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "videoprism_source_commit": SOURCE_COMMIT,
        "input_shape": list(video.shape),
        "token_shape": list(tokens.shape),
        "token_grid_shape": [1, 8, 16, 16, 768],
        "sector_feature_shape": list(sectors.shape),
        "sector_order": [
            "front",
            "front_left",
            "left",
            "back_left",
            "back",
            "back_right",
            "right",
            "front_right",
        ],
        "sector_patch_columns": [
            [7, 8],
            [5, 6],
            [3, 4],
            [1, 2],
            [15, 0],
            [13, 14],
            [11, 12],
            [9, 10],
        ],
        "preprocessing": "direct resize of full equirectangular frames to 288x288; no crop",
        "pooled_reference_id": pooled_entry["id"],
        "pooled_reference_max_abs_diff": pooled_max_abs_diff,
        "sector_file": sector_path.name,
        "sector_file_sha256": sha256_file(sector_path),
        "sector_file_bytes": sector_path.stat().st_size,
        "wall_seconds": time.perf_counter() - started,
        "training": False,
        "gpu_used": False,
        "calibration_validation_dev_test_or_audit_labels_read": False,
        "note": (
            "This validates token-grid extraction and pooled-feature parity. The "
            "sector column mapping follows the STARSS23 front-zero/positive-left "
            "azimuth convention over the equirectangular image; it does not prove "
            "that a particular source is visually detectable in the selected window."
        ),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
