"""Freeze a freeform-validation anomaly threshold before opening probe payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--predictor-run-metadata", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if str(SOURCE_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))
    from scripts.evaluate_physical_concepts_probe import (
        _sequence_scores,
        fixed_validation_threshold,
    )
    from scripts.train_intphys_next_feature_predictor import load_feature_split
    from tiny_omni_decision.temporal_prediction import NextFrameFeaturePredictor

    args = _args()
    manifest_path = args.manifest.resolve(strict=True)
    checkpoint_path = args.predictor_checkpoint.resolve(strict=True)
    metadata_path = args.predictor_run_metadata.resolve(strict=True)
    validation_path = args.validation_features.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite threshold receipt: {output_path}")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or torch.cuda.device_count() != 0
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("threshold freezing must remain CPU-only")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = manifest["candidate"]
    if manifest.get("status") != "FROZEN_BEFORE_PROBE_PAYLOAD_READ":
        raise ValueError("probe scoring/sample protocol is not frozen")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        sha256_file(checkpoint_path) != candidate["predictor_checkpoint_sha256"]
        or sha256_file(metadata_path) != candidate["predictor_run_metadata_sha256"]
        or sha256_file(validation_path) != candidate["validation_feature_archive_sha256"]
        or metadata.get("best_checkpoint_sha256") != candidate["predictor_checkpoint_sha256"]
        or metadata.get("validation_features_sha256")
        != candidate["validation_feature_archive_sha256"]
        or metadata.get("status") != "complete_physical_concepts_frozen_feature_next_step_fit"
    ):
        raise ValueError("selected checkpoint/metadata/validation differ from frozen hashes")

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    validation, validation_ids, _ = load_feature_split(validation_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = NextFrameFeaturePredictor(feature_width=int(state["feature_width"]), hidden_width=64)
    model.load_state_dict(state["state_dict"], strict=True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    errors = _sequence_scores(model, validation)
    mean_loss = float(errors.mean())
    if abs(mean_loss - metadata["reloaded_validation_cosine_loss"]) > 1e-7:
        raise ValueError("selected checkpoint validation replay differs from saved run metadata")
    threshold, scale = fixed_validation_threshold(errors)

    receipt = {
        "status": "frozen_before_probe_payload_download",
        "started_at_utc": started_at.isoformat(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "device": "cpu",
        "gpu_compute_used": False,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "predictor_checkpoint_sha256": candidate["predictor_checkpoint_sha256"],
        "predictor_metadata_sha256": candidate["predictor_run_metadata_sha256"],
        "validation_feature_archive_sha256": candidate["validation_feature_archive_sha256"],
        "validation_sample_count": len(validation_ids),
        "validation_sample_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "validation_mean_cosine_loss": mean_loss,
        "validation_q95_anomaly_threshold": threshold,
        "validation_mad_scale": scale,
        "probability_calibration": "none; one-class anomaly diagnostic only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
