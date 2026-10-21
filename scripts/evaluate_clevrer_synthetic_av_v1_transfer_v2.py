"""Evaluate frozen v1 readout checkpoints on the new task-specific v2 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scripts.train_clevrer_synthetic_av_temporal_decision import (
    _load_inputs,
    _metrics,
    _model_logits,
    _prediction_rows,
    _scene_pair_metrics,
    _validate_experiment_config,
)
from tiny_omni_decision.temporal_decision import CausalGatedTemporalDecision

V1_EXPERIMENT_ID = "clevrer-synthetic-av-temporal-decision-v1"
CANDIDATES = {
    "joint_lstm": "joint",
    "audio_only_lstm": "audio_only",
    "video_only_lstm": "video_only",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_transfer(
    *,
    config_path: Path,
    feature_dir: Path,
    feature_report_path: Path,
    sample_dir: Path,
    v1_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite transfer evaluation: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_experiment_config(config)
    if config.get("experiment_id") != "clevrer-synthetic-av-temporal-decision-v2":
        raise ValueError("transfer evaluation requires the frozen v2 data configuration")
    rows_train, rows_validation, tensors, input_report = _load_inputs(
        config, feature_dir, feature_report_path, sample_dir
    )
    del rows_train
    train_count = int(input_report["split"]["train_count"])
    validation_targets = tensors["targets"][train_count:]
    device = torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_results = []
    for name, mode in CANDIDATES.items():
        checkpoint_path = v1_run_dir / f"best-{name}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("candidate") != name or checkpoint.get("selector") != (
            "minimum validation NLL"
        ):
            raise ValueError(f"unexpected v1 checkpoint metadata for {name}")
        checkpoint_config = checkpoint.get("config", {})
        _validate_experiment_config(checkpoint_config)
        if checkpoint_config.get("experiment_id") != V1_EXPERIMENT_ID:
            raise ValueError(f"checkpoint is not from the frozen v1 run: {name}")
        model = CausalGatedTemporalDecision(
            video_size=int(config["readout"]["video_size"]),
            audio_size=int(config["readout"]["audio_size"]),
            text_size=int(config["readout"]["text_size"]),
            hidden_size=int(config["readout"]["hidden_size"]),
            modality_mode=mode,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        with torch.inference_mode():
            logits = _model_logits(
                model,
                tensors["video"][train_count:],
                tensors["audio"][train_count:],
                tensors["text"][train_count:],
            ).float()
        metrics = _metrics(logits, validation_targets)
        pairs = _scene_pair_metrics(rows_validation, logits, validation_targets)
        predictions = _prediction_rows(rows_validation, logits, validation_targets)
        prediction_path = output_dir / f"predictions-{name}.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
            encoding="utf-8",
        )
        candidate_results.append(
            {
                "candidate": name,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "v1_selected_step": checkpoint.get("step"),
                "v2_validation": metrics,
                "v2_validation_pairs": pairs,
                "predictions_file": prediction_path.name,
                "predictions_sha256": _sha256(prediction_path),
            }
        )
    report = {
        "status": "complete_v1_transfer_evaluation_on_v2_task_specific_validation",
        "v1_run_metadata_sha256": _sha256(v1_run_dir / "run-metadata.json"),
        "v1_results_sha256": _sha256(v1_run_dir / "results.json"),
        "config_sha256": _sha256(config_path),
        "feature_report_sha256": _sha256(feature_report_path),
        "sample_manifest_sha256": input_report["split"]["sample_manifest_sha256"],
        "validation_id_order_sha256": input_report["split"]["validation_id_order_sha256"],
        "validation_scene_count": len({row["scene_group_id"] for row in rows_validation}),
        "validation_sample_count": len(rows_validation),
        "candidate_results": candidate_results,
        "checkpoint_selection_used_v2_validation": False,
        "validation_is_task_specific_not_media_unseen_or_blind": True,
        "sealed_or_official_validation_test_opened": False,
    }
    report_path = output_dir / "transfer-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--v1-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_transfer(
        config_path=args.config.resolve(strict=True),
        feature_dir=args.feature_dir.resolve(strict=True),
        feature_report_path=args.feature_report.resolve(strict=True),
        sample_dir=args.sample_dir.resolve(strict=True),
        v1_run_dir=args.v1_run_dir.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
