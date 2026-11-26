"""Recheck saved MiniWoB checkpoints and finalize the held-out option-order control."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from scripts.miniwob_screen_action_data import output_option_ids
from scripts.train_miniwob_dino_path_a import read_corpus, targets_for
from tiny_omni_decision.miniwob_dino_readout import (
    MiniWoBOptionScorer,
    classification_metrics,
    permute_candidate_axis,
)

OUTPUT = Path("D:/CodexArtifacts/pretrained-reuse-r2/miniwob-dino-path-a-v1-attempt-01")
OPTION_ORDER_TOLERANCE = 2e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(2)
    config = json.loads((OUTPUT / "effective-config.json").read_text(encoding="utf-8"))
    if config.get("cuda_used") is not False or config["device"] != "cpu":
        raise ValueError("saved run is not the expected CPU-only experiment")
    corpus_root, _, splits = read_corpus(config)
    val_rows = splits["validation"]
    val_targets = targets_for(val_rows)
    with np.load(OUTPUT / "frozen-features.npz", allow_pickle=False) as archive:
        features = torch.from_numpy(archive["validation"].copy())
        targets = torch.from_numpy(archive["validation_targets"].copy())
    if (
        features.shape != (len(val_rows), 4, 1154)
        or not torch.equal(targets, val_targets)
        or not torch.isfinite(features).all()
    ):
        raise ValueError("saved validation features or target order differ from frozen corpus")
    model = MiniWoBOptionScorer().cpu()
    checkpoint = OUTPUT / "matched_screen" / "checkpoint-best.safetensors"
    model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    model.eval()
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_features, inverse = permute_candidate_axis(features, permutation)
    with torch.inference_mode():
        original = torch.softmax(model(features), dim=1)
        permuted = torch.softmax(model(permuted_features), dim=1).index_select(1, inverse)
    differences = (original - permuted).abs()
    max_difference = float(differences.max())
    argmax_changes = int((original.argmax(dim=1) != permuted.argmax(dim=1)).sum())
    if max_difference > OPTION_ORDER_TOLERANCE or argmax_changes:
        raise RuntimeError("saved checkpoint failed candidate-order ID-alignment recheck")

    saved_records = [
        json.loads(line)
        for line in (OUTPUT / "matched_screen" / "validation-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(saved_records) != len(val_rows):
        raise ValueError("saved validation prediction count mismatch")
    saved_probabilities: list[list[float]] = []
    for row, saved in zip(val_rows, saved_records, strict=True):
        ids = output_option_ids(row)
        if saved["sample_id"] != row["sample_id"]:
            raise ValueError("saved prediction sample order differs from frozen validation order")
        if saved["target_option_id"] != ids[row["target_index"]]:
            raise ValueError("saved target option ID differs from frozen row")
        saved_probabilities.append(
            [saved["probabilities_by_option_id"][option_id] for option_id in ids]
        )
    saved_tensor = torch.tensor(saved_probabilities, dtype=torch.float32)
    saved_replay_delta = float((saved_tensor - original).abs().max())
    if saved_replay_delta != 0:
        raise RuntimeError("saved selected-checkpoint probabilities do not replay exactly")

    arm_metrics = {}
    total_fit_seconds = 0.0
    selected_epochs = {}
    for arm in ("matched_screen", "zeroed_screen", "permuted_screen"):
        arm_root = OUTPUT / arm
        metric = json.loads((arm_root / "metrics.json").read_text(encoding="utf-8"))
        history = [
            json.loads(line)
            for line in (arm_root / "training-history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if (
            not history
            or metric["selected_checkpoint"]["best_reload_max_probability_abs_difference"] != 0
        ):
            raise ValueError(f"saved training history/reload check invalid for {arm}")
        total_fit_seconds += sum(float(row["seconds"]) for row in history)
        selected_epochs[arm] = metric["selected_checkpoint"]["epoch"]
        arm_metrics[arm] = metric

    prior = json.loads((OUTPUT / "position-prior-metrics.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS_EXECUTION_POSTHOC_FINALIZED",
        "task_scope": "synthetic MiniWoB++ click-color screen-only development task",
        "validation_is_development_only": True,
        "not_multimodal_product_evidence": True,
        "training_artifacts_exit": (
            "all three arms reached their recorded terminal epochs and saved best/final "
            "checkpoints; process initially exited at the post-fit numerical-tolerance assertion"
        ),
        "corpus": {
            "root": str(corpus_root),
            "train_count": len(splits["train"]),
            "validation_count": len(val_rows),
            "train_jsonl_sha256": config["corpus"]["train_jsonl_sha256"],
            "validation_jsonl_sha256": config["corpus"]["validation_jsonl_sha256"],
            "manifest_sha256": config["corpus"]["manifest_sha256"],
            "sample_order_hashes": {
                "train": config["train_sample_order_sha256"],
                "validation": config["validation_sample_order_sha256"],
            },
        },
        "selected_rule": config["effective_readout"]["selector"],
        "training": {
            "selected_epochs": selected_epochs,
            "sum_of_recorded_fit_epoch_seconds": total_fit_seconds,
            "feature_extraction_seconds": config["feature_extraction_seconds"],
            "cuda_used": False,
            "cloud_cost_usd": 0,
        },
        "position_prior": prior,
        "arms": arm_metrics,
        "candidate_order_permutation": {
            "input_permutation": permutation.tolist(),
            "max_option_aligned_probability_abs_difference": max_difference,
            "mean_option_aligned_probability_abs_difference": float(differences.mean()),
            "p99_option_aligned_probability_abs_difference": float(
                torch.quantile(differences.flatten(), 0.99)
            ),
            "tolerance": OPTION_ORDER_TOLERANCE,
            "argmax_changes_after_ID_remap": argmax_changes,
            "accuracy_after_ID_remap": float(
                classification_metrics(permuted, val_targets)["accuracy"]
            ),
            "matched_accuracy": float(classification_metrics(original, val_targets)["accuracy"]),
            "saved_prediction_replay_max_abs_difference": saved_replay_delta,
            "status": "PASS",
        },
        "parameters": {
            "frozen_dinov2": config["encoder"]["parameters"],
            "trained_readout": arm_metrics["matched_screen"]["selected_checkpoint"][
                "trainable_readout_parameters"
            ],
            "resident_total": config["encoder"]["parameters"]
            + arm_metrics["matched_screen"]["selected_checkpoint"]["trainable_readout_parameters"],
            "updated_encoder_parameters": 0,
        },
        "checkpoint_hashes": {
            arm: {
                "best": arm_metrics[arm]["selected_checkpoint"]["checkpoint_sha256"],
                "final": arm_metrics[arm]["selected_checkpoint"]["final_checkpoint_sha256"],
            }
            for arm in arm_metrics
        },
        "finalized_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_path = OUTPUT / "experiment-results.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        result["finalized_at_local"] = existing.get("finalized_at_local")
        if existing != result:
            differing_keys = [key for key in result if existing.get(key) != result[key]]
            raise ValueError(
                "saved finalized result differs from deterministic recheck: "
                + ", ".join(differing_keys)
            )
        recheck_path = OUTPUT / "candidate-order-recheck.json"
        recheck = json.loads(recheck_path.read_text(encoding="utf-8"))
        if recheck != result["candidate_order_permutation"]:
            raise ValueError("saved option-order artifact differs from deterministic recheck")
        print(json.dumps({"status": "EXISTING_RESULT_VERIFIED", "result": result}, indent=2))
        return
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "candidate-order-recheck.json").write_text(
        json.dumps(result["candidate_order_permutation"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
