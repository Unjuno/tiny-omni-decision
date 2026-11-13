"""Train one fixed conditional-gating readout on frozen AV features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.fit_perception_temporal_match_readouts import (
    categorical_metrics,
    load_features,
    paired_bootstrap_delta,
    sha256_file,
)

SEED = 17
HIDDEN_DIMENSION = 16
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-3
TOTAL_UPDATES = 2048
EVALUATION_INTERVAL = 32


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    manifest_path = args.manifest.resolve(strict=True)
    video_result_path = args.video_features.resolve(strict=True)
    audio_result_path = args.audio_features.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_result = json.loads(video_result_path.read_text(encoding="utf-8"))
    audio_result = json.loads(audio_result_path.read_text(encoding="utf-8"))
    if video_result.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("VideoPrism feature manifest mismatch")
    if audio_result.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("EfficientAT feature manifest mismatch")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gated-readout result: {output}")
    output.mkdir(parents=True)
    samples = load_features(manifest, video_result, audio_result)
    train_rows = [samples[row["sample_id"]] for row in manifest["train"]]
    val_rows = [samples[row["sample_id"]] for row in manifest["validation"]]
    if {row["row"]["video_id"] for row in train_rows} & {
        row["row"]["video_id"] for row in val_rows
    }:
        raise ValueError("train and validation video groups overlap")

    train_video = np.repeat(np.stack([row["video"] for row in train_rows]), 2, axis=0)
    train_audio = np.concatenate([row["audio"] for row in train_rows], axis=0)
    val_video = np.repeat(np.stack([row["video"] for row in val_rows]), 2, axis=0)
    val_audio = np.concatenate([row["audio"] for row in val_rows], axis=0)
    train_target = torch.tensor(
        [row["row"]["target_index"] for row in train_rows], dtype=torch.long
    )
    val_target = torch.tensor([row["row"]["target_index"] for row in val_rows], dtype=torch.long)

    video_mean = torch.tensor(train_video.mean(axis=0), dtype=torch.float32)
    video_scale = torch.tensor(train_video.std(axis=0) + 1e-5, dtype=torch.float32)
    audio_mean = torch.tensor(train_audio.mean(axis=0), dtype=torch.float32)
    audio_scale = torch.tensor(train_audio.std(axis=0) + 1e-5, dtype=torch.float32)
    train_v = (torch.tensor(train_video) - video_mean) / video_scale
    train_a = (torch.tensor(train_audio) - audio_mean) / audio_scale
    val_v = (torch.tensor(val_video) - video_mean) / video_scale
    val_a = (torch.tensor(val_audio) - audio_mean) / audio_scale
    train_v = train_v.reshape(len(train_rows), 2, -1)
    train_a = train_a.reshape(len(train_rows), 2, -1)
    val_v = val_v.reshape(len(val_rows), 2, -1)
    val_a = val_a.reshape(len(val_rows), 2, -1)

    class ConditionalGate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.video_projection = torch.nn.Linear(768, HIDDEN_DIMENSION)
            self.audio_projection = torch.nn.Linear(384, HIDDEN_DIMENSION)
            self.gate = torch.nn.Linear(HIDDEN_DIMENSION * 2, HIDDEN_DIMENSION)
            self.score = torch.nn.Linear(HIDDEN_DIMENSION, 1)

        def forward(self, video, audio, *, return_gate: bool = False):
            video_value = torch.tanh(self.video_projection(video))
            audio_value = torch.tanh(self.audio_projection(audio))
            gate_value = torch.sigmoid(self.gate(torch.cat([video_value, audio_value], dim=-1)))
            fused = gate_value * video_value + (1.0 - gate_value) * audio_value
            scores = self.score(fused).squeeze(-1)
            if return_gate:
                return scores, gate_value
            return scores

    model = ConditionalGate().cpu()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != 19_009:
        raise ValueError(f"unexpected gated-readout parameter count: {trainable_parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    best = {"step": None, "validation_nll": float("inf"), "state": None, "scores": None}
    started = time.perf_counter()

    def metrics_at(step: int):
        model.eval()
        with torch.inference_mode():
            train_scores = model(train_v, train_a).numpy()
            val_scores, val_gate = model(val_v, val_a, return_gate=True)
            val_scores = val_scores.numpy()
            gate_values = val_gate.numpy()
        train_metrics = categorical_metrics(train_scores, train_target.numpy())
        val_metrics = categorical_metrics(val_scores, val_target.numpy())
        record = {
            "step": step,
            "train_loss": float(
                torch.nn.functional.cross_entropy(
                    torch.as_tensor(train_scores, dtype=torch.float32), train_target
                )
            ),
            "validation_nll": val_metrics["nll"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_brier": val_metrics["brier"],
            "validation_ece_10_equal_width": val_metrics["ece_10_equal_width"],
            "mean_gate": float(gate_values.mean()),
            "gate_std": float(gate_values.std()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if record["validation_nll"] < best["validation_nll"]:
            best.update(
                step=step,
                validation_nll=record["validation_nll"],
                state={key: value.detach().clone() for key, value in model.state_dict().items()},
                scores=val_scores.copy(),
            )
        history.append(record)
        with (output / "training-history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return record

    metrics_at(0)
    for step in range(1, TOTAL_UPDATES + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(train_v, train_a)
        loss = torch.nn.functional.cross_entropy(scores, train_target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at update {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at update {step}")
        optimizer.step()
        if step % EVALUATION_INTERVAL == 0 or step == TOTAL_UPDATES:
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError(f"non-finite model parameter at update {step}")
            row = metrics_at(step)
            row["gradient_norm"] = float(gradient_norm)
            row["learning_rate"] = LEARNING_RATE
            if step % 256 == 0:
                print(json.dumps(row), flush=True)

    if best["state"] is None:
        raise RuntimeError("validation selector did not select a checkpoint")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": best["state"],
            "seed": SEED,
            "selected_step": best["step"],
            "selection_rule": "minimum validation categorical NLL; ties keep earlier step",
            "architecture": "16-dimensional conditional gate over pooled AV features",
        },
        output / "best-checkpoint.pt",
    )
    torch.save(
        {
            "state_dict": final_state,
            "seed": SEED,
            "step": TOTAL_UPDATES,
            "architecture": "16-dimensional conditional gate over pooled AV features",
        },
        output / "final-checkpoint.pt",
    )
    model.load_state_dict(best["state"], strict=True)
    model.eval()
    with torch.inference_mode():
        best_scores, best_gate = model(val_v, val_a, return_gate=True)
    best_scores = best_scores.numpy()
    best_metrics = categorical_metrics(best_scores, val_target.numpy())
    if abs(best_metrics["nll"] - best["validation_nll"]) > 1e-10:
        raise RuntimeError("best-checkpoint reload does not reproduce selected validation NLL")
    audio_missing = torch.zeros_like(val_a)
    video_missing = torch.zeros_like(val_v)
    with torch.inference_mode():
        audio_missing_scores = model(val_v, audio_missing).numpy()
        video_missing_scores = model(video_missing, val_a).numpy()
        video_shuffled = val_v.clone()
        audio_shuffled = val_a.clone()
        rng = np.random.default_rng(SEED)
        label_ids = np.asarray([row["row"]["label_id"] for row in val_rows])
        shuffled_sources = np.arange(len(val_rows))
        singleton_groups = 0
        for label_id in np.unique(label_ids):
            indexes = np.flatnonzero(label_ids == label_id)
            if len(indexes) < 2:
                singleton_groups += len(indexes)
                continue
            offset = int(rng.integers(1, len(indexes)))
            shuffled_sources[indexes] = np.roll(indexes, offset)
        video_shuffled = video_shuffled[shuffled_sources]
        audio_shuffled = audio_shuffled[shuffled_sources]
        video_shuffled_scores = model(video_shuffled, val_a).numpy()
        audio_shuffled_scores = model(val_v, audio_shuffled).numpy()
    target_np = val_target.numpy()
    best_correct = best_scores.argmax(axis=1) == target_np
    if abs(best_metrics["accuracy"] - float(best_correct.mean())) > 1e-12:
        raise RuntimeError("best selected accuracy is inconsistent")
    baseline_path = args.baseline_result.resolve(strict=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_audio = np.asarray(
        [np.asarray(row["audio_only_scores"], dtype=np.float64) for row in baseline["predictions"]]
    )
    if [row["sample_id"] for row in baseline["predictions"]] != [
        row["row"]["sample_id"] for row in val_rows
    ]:
        raise ValueError("baseline predictions do not match the frozen validation order")
    audio_only_correct = baseline_audio.argmax(axis=1) == target_np
    paired = paired_bootstrap_delta(best_correct, audio_only_correct, seed=SEED)
    predictions = []
    for index, item in enumerate(val_rows):
        predictions.append(
            {
                "sample_id": item["row"]["sample_id"],
                "video_id": item["row"]["video_id"],
                "target_index": int(target_np[index]),
                "scores": best_scores[index].tolist(),
                "probabilities": best_metrics["probabilities"][index].tolist(),
                "prediction": int(best_metrics["predictions"][index]),
                "gate_mean_for_candidate_options": best_gate[index].mean(dim=-1).tolist(),
            }
        )
    result = {
        "status": "PASS_CONDITIONAL_GATE_READOUT",
        "manifest_sha256": sha256_file(manifest_path),
        "video_features_sha256": sha256_file(video_result_path),
        "audio_features_sha256": sha256_file(audio_result_path),
        "baseline_result_sha256": sha256_file(baseline_path),
        "seed": SEED,
        "train_videos": len(train_rows),
        "validation_videos": len(val_rows),
        "trainable_parameters": trainable_parameters,
        "architecture": (
            "video/audio linear projections to 16; sigmoid conditional gate; scalar score"
        ),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_updates": TOTAL_UPDATES,
        "validation_interval": EVALUATION_INTERVAL,
        "selector": "minimum validation categorical NLL; ties keep earlier step",
        "selected_step": best["step"],
        "best_checkpoint_sha256": sha256_file(output / "best-checkpoint.pt"),
        "final_checkpoint_sha256": sha256_file(output / "final-checkpoint.pt"),
        "selected_validation_metrics": {
            key: value
            for key, value in best_metrics.items()
            if key not in ("probabilities", "predictions")
        },
        "final_validation_metrics": {
            "accuracy": history[-1]["validation_accuracy"],
            "nll": history[-1]["validation_nll"],
            "brier": history[-1]["validation_brier"],
            "ece_10_equal_width": history[-1]["validation_ece_10_equal_width"],
        },
        "train_history": history,
        "controls": {
            "video_missing_mean_imputed": {
                key: value
                for key, value in categorical_metrics(video_missing_scores, target_np).items()
                if key not in ("probabilities", "predictions")
            },
            "audio_missing_mean_imputed": {
                key: value
                for key, value in categorical_metrics(audio_missing_scores, target_np).items()
                if key not in ("probabilities", "predictions")
            },
            "video_shuffled_within_sound_label": {
                key: value
                for key, value in categorical_metrics(video_shuffled_scores, target_np).items()
                if key not in ("probabilities", "predictions")
            },
            "audio_shuffled_within_sound_label": {
                key: value
                for key, value in categorical_metrics(audio_shuffled_scores, target_np).items()
                if key not in ("probabilities", "predictions")
            },
            "singleton_label_video_count_not_shufflable": singleton_groups,
        },
        "paired_selected_gate_minus_audio_only_accuracy": paired,
        "selected_vs_baseline_joint_accuracy_delta": best_metrics["accuracy"]
        - baseline["validation_metrics"]["joint"]["accuracy"],
        "validation_scope": (
            "development-only; checkpoint selected on this validation; no sealed audit"
        ),
        "predictions": predictions,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {
        key: value for key, value in result.items() if key not in ("predictions", "train_history")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
