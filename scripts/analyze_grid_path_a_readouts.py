"""Recheck Path A readout checkpoints and paired speaker-cluster sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.fit_grid_path_a_readouts import (
    ARRAY_HASHES,
    COHORT,
    FEATURES,
    SlotReadout,
    sha,
    validate_inputs,
)
from scripts.grid_path_a_data import OPTIONS

FEATURE_RESULT_SHA = "40a06e5b4a3c69b6c965a48b59d4f08a32000f7b25c0a11cd1038f37af2e87e3"
READOUT_RESULT_SHA = "449e2ea2e292842c9f7bcf78e02a4dbaece6916c3f3026996a2d8511f147316c"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 1701


def metrics(probabilities: list[np.ndarray], targets: np.ndarray, names: list[str]):
    by_slot = {}
    for slot, name in enumerate(names):
        p = probabilities[slot]
        target = targets[:, slot]
        one_hot = np.eye(p.shape[1], dtype=np.float32)[target]
        prediction = p.argmax(-1)
        confidence = p.max(-1)
        correct = (prediction == target).astype(np.float32)
        ece = 0.0
        for index in range(10):
            lower, upper = index / 10, (index + 1) / 10
            chosen = (confidence >= lower) & (
                confidence < upper if index < 9 else confidence <= upper
            )
            if chosen.any():
                ece += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
        by_slot[name] = {
            "accuracy": float(correct.mean()),
            "nll": float(-np.log(np.clip(p[np.arange(len(target)), target], 1e-12, 1)).mean()),
            "brier": float(np.square(p - one_hot).sum(-1).mean()),
            "ece_10bin": float(ece),
            "count": len(target),
        }
    return {
        "slot_metrics": by_slot,
        "macro_accuracy": float(np.mean([x["accuracy"] for x in by_slot.values()])),
        "macro_nll": float(np.mean([x["nll"] for x in by_slot.values()])),
        "macro_brier": float(np.mean([x["brier"] for x in by_slot.values()])),
        "macro_ece": float(np.mean([x["ece_10bin"] for x in by_slot.values()])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_dir() or (run / "analysis.json").exists():
        raise ValueError("existing complete run and new analysis path required")
    audio, video, targets, ids, cohort, classes = validate_inputs()
    if sha(run / "result.json") != READOUT_RESULT_SHA:
        raise ValueError("readout result hash differs")
    config = json.loads((run / "effective-config.json").read_text(encoding="utf-8"))
    if (
        config.get("feature_result_sha256") != FEATURE_RESULT_SHA
        or config.get("feature_hashes") != ARRAY_HASHES
    ):
        raise ValueError("readout run is linked to unexpected frozen features")
    with np.load(run / "train-only-normalizer.npz", allow_pickle=False) as data:
        a_train = ((audio[:768] - data["audio_mean"]) / data["audio_scale"]).astype(np.float32)
        a_val = ((audio[768:] - data["audio_mean"]) / data["audio_scale"]).astype(np.float32)
        v_train = ((video[:768] - data["video_mean"]) / data["video_scale"]).astype(np.float32)
        v_val = ((video[768:] - data["video_mean"]) / data["video_scale"]).astype(np.float32)
    masks = {"audio": [1.0, 0.0], "video": [0.0, 1.0], "joint": [1.0, 1.0]}
    output = {
        "status": "PASS_GRID_PATH_A_READOUT_RECHECK",
        "train_metrics_fixed_checkpoint": {},
        "validation_metrics_recomputed": {},
        "speaker_cluster_paired_delta": {},
        "prediction_replay_max_abs_probability_delta": {},
        "bootstrap": {
            "unit": "validation speaker group",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 2.5%-97.5%",
        },
        "limitations": [
            "Only eight held-out speakers; cluster bootstrap is a coarse diagnostic.",
            "Validation selected checkpoints at every epoch, so intervals are not blind "
            "confirmatory inference.",
            "This is a closed-vocabulary synchronized speech-slot task, not general "
            "multimodal reasoning.",
        ],
    }
    saved_predictions = {}
    for arm in ("audio", "video", "joint"):
        model = SlotReadout(classes).cpu()
        model.load_state_dict(
            torch.load(run / f"{arm}-best.pt", map_location="cpu", weights_only=True), strict=True
        )
        with torch.inference_mode():
            train_logits = model(
                torch.from_numpy(a_train),
                torch.from_numpy(v_train),
                torch.tensor(masks[arm]).repeat(768, 1),
            )
            val_logits = model(
                torch.from_numpy(a_val),
                torch.from_numpy(v_val),
                torch.tensor(masks[arm]).repeat(256, 1),
            )
        train_probs = [x.softmax(-1).numpy() for x in train_logits]
        val_probs = [x.softmax(-1).numpy() for x in val_logits]
        output["train_metrics_fixed_checkpoint"][arm] = metrics(
            train_probs, targets[:768], list(OPTIONS)
        )
        val_metrics = metrics(val_probs, targets[768:], list(OPTIONS))
        output["validation_metrics_recomputed"][arm] = val_metrics
        saved_predictions[arm] = val_probs
        with np.load(run / f"{arm}-validation-predictions.npz", allow_pickle=False) as saved:
            if saved["source_ids"].tolist() != ids[768:] or not np.array_equal(
                saved["targets"], targets[768:]
            ):
                raise ValueError(f"saved {arm} prediction identities/targets differ")
            for slot, probabilities in enumerate(val_probs):
                delta = float(np.max(np.abs(saved[f"prob_{slot}"] - probabilities)))
                output["prediction_replay_max_abs_probability_delta"][f"{arm}:{slot}"] = delta
                if delta > 1e-6 or not np.array_equal(
                    saved[f"prediction_{slot}"], probabilities.argmax(-1)
                ):
                    raise ValueError(f"saved {arm} validation predictions do not reload")
    with np.load(run / "joint-mismatched-video-predictions.npz", allow_pickle=False) as saved:
        if saved["source_ids"].tolist() != ids[768:] or not np.array_equal(
            saved["targets"], targets[768:]
        ):
            raise ValueError("mismatch prediction identity/target mismatch")
        mismatch_probs = [saved[f"prob_{i}"] for i in range(6)]
        mismatch_ids = saved["donor_ids"].tolist()
    output["validation_metrics_recomputed"]["joint_mismatched_video"] = metrics(
        mismatch_probs, targets[768:], list(OPTIONS)
    )
    if any(
        donor.split("_", 1)[0] == source.split("_", 1)[0]
        for donor, source in zip(mismatch_ids, ids[768:], strict=True)
    ):
        raise ValueError("video donor not from another held-out speaker")
    speaker_ids = [asset.split("_", 1)[0] for asset in ids[768:]]
    speakers = cohort["validation_speakers"]
    matched_correct = np.stack(
        [saved_predictions["joint"][i].argmax(-1) == targets[768:, i] for i in range(6)]
    ).mean(0)
    mismatch_correct = np.stack(
        [mismatch_probs[i].argmax(-1) == targets[768:, i] for i in range(6)]
    ).mean(0)
    paired = {}
    for speaker in speakers:
        mask = np.asarray([value == speaker for value in speaker_ids])
        paired[speaker] = float((matched_correct - mismatch_correct)[mask].mean())
    values = np.asarray(list(paired.values()), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(
        axis=1
    )
    output["speaker_cluster_paired_delta"] = {
        "definition": (
            "joint matched-video macro slot Accuracy minus cross-speaker-mismatched-video "
            "macro slot Accuracy"
        ),
        "speaker_deltas": paired,
        "mean_delta": float(values.mean()),
        "percentile_95_interval": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }
    output["run_result_sha256"] = sha(run / "result.json")
    output["checkpoint_sha256"] = {
        name: sha(run / f"{name}-best.pt") for name in ("audio", "video", "joint")
    }
    output["feature_result_sha256"] = sha(FEATURES / "result.json")
    output["cohort_sha256"] = sha(COHORT)
    (run / "analysis.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "macro_accuracy": {
                    k: v["macro_accuracy"]
                    for k, v in output["validation_metrics_recomputed"].items()
                },
                "paired_delta": output["speaker_cluster_paired_delta"],
                "analysis_sha256": sha(run / "analysis.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
