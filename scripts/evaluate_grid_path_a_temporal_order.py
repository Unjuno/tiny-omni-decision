"""Measure temporal-order sensitivity of the fixed GRID Path A readouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_grid_path_a_readouts import metrics
from scripts.fit_grid_path_a_readouts import (
    ARRAY_HASHES,
    SlotReadout,
    sha,
    validate_inputs,
)
from scripts.grid_path_a_data import OPTIONS

RUN = Path("C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-readouts-20261028-attempt-01")
RUN_RESULT_SHA = "449e2ea2e292842c9f7bcf78e02a4dbaece6916c3f3026996a2d8511f147316c"
PERMUTATION_SEED = 1702


def predict(model, audio, video, mask):
    model.eval()
    with torch.inference_mode():
        return [value.softmax(-1).numpy() for value in model(audio, video, mask)]


def main():
    destination = RUN / "temporal-order-control.json"
    predictions_path = RUN / "temporal-order-control-predictions.npz"
    if destination.exists() or predictions_path.exists():
        raise ValueError("temporal-order control already exists; preserve its outputs")
    if sha(RUN / "result.json") != RUN_RESULT_SHA:
        raise ValueError("fixed readout run result changed")
    audio, video, targets, ids, cohort, classes = validate_inputs()
    with np.load(RUN / "train-only-normalizer.npz", allow_pickle=False) as data:
        audio = ((audio[768:] - data["audio_mean"]) / data["audio_scale"]).astype(np.float32)
        video = ((video[768:] - data["video_mean"]) / data["video_scale"]).astype(np.float32)
    target = targets[768:]
    audio_t = torch.from_numpy(audio)
    original_video_t = torch.from_numpy(video.copy())
    rng = np.random.default_rng(PERMUTATION_SEED)
    random_order = np.stack([rng.permutation(16) for _ in range(len(video))])
    variants = {
        "chronological": np.broadcast_to(np.arange(16), (len(video), 16)),
        "reversed": np.broadcast_to(np.arange(15, -1, -1), (len(video), 16)),
        "independently_shuffled": random_order,
    }
    output = {
        "status": "PASS_GRID_PATH_A_TEMPORAL_ORDER_CONTROL",
        "seed": PERMUTATION_SEED,
        "validation_rows": len(video),
        "validation_ids_sha256": __import__("hashlib")
        .sha256("\n".join(ids[768:]).encode())
        .hexdigest(),
        "feature_hashes": ARRAY_HASHES,
        "permutations": {name: value.tolist() for name, value in variants.items()},
        "metrics": {},
        "prediction_change_fraction_vs_chronological": {},
        "limitations": [
            "All measurements reuse the same development validation cohort used for "
            "checkpoint selection.",
            "Temporal permutation is an input sensitivity diagnostic, not proof that "
            "order encodes useful semantics.",
            "Only eight independent validation speakers are represented.",
        ],
    }
    saved = {"source_ids": np.asarray(ids[768:]), "targets": target}
    for arm in ("video", "joint"):
        model = SlotReadout(classes).cpu()
        model.load_state_dict(
            torch.load(RUN / f"{arm}-best.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        mask = torch.tensor([0.0, 1.0] if arm == "video" else [1.0, 1.0]).repeat(len(video), 1)
        chronological = None
        for variant, order in variants.items():
            video_t = original_video_t
            if variant != "chronological":
                video_t = torch.from_numpy(
                    np.take_along_axis(video, order[:, :, None], axis=1).copy()
                )
            probabilities = predict(model, audio_t, video_t, mask)
            report = metrics(probabilities, target, list(OPTIONS))
            predictions = [value.argmax(-1) for value in probabilities]
            if variant == "chronological":
                chronological = predictions
            else:
                changed = np.concatenate(
                    [predictions[index] != chronological[index] for index in range(6)]
                )
                output["prediction_change_fraction_vs_chronological"][f"{arm}:{variant}"] = float(
                    changed.mean()
                )
            output["metrics"][f"{arm}:{variant}"] = {
                key: value for key, value in report.items() if key != "slot_metrics"
            }
            output["metrics"][f"{arm}:{variant}"]["slot_metrics"] = report["slot_metrics"]
            for slot, probability in enumerate(probabilities):
                saved[f"{arm}_{variant}_prob_{slot}"] = probability
                saved[f"{arm}_{variant}_prediction_{slot}"] = predictions[slot]
    np.savez_compressed(predictions_path, **saved)
    output["run_result_sha256"] = sha(RUN / "result.json")
    output["prediction_file_sha256"] = sha(predictions_path)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "macro_accuracy": {
                    name: value["macro_accuracy"] for name, value in output["metrics"].items()
                },
                "prediction_change_fraction_vs_chronological": output[
                    "prediction_change_fraction_vs_chronological"
                ],
                "report_sha256": sha(destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
