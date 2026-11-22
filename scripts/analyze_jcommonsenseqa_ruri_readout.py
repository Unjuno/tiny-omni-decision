"""Compare frozen Ruri cosine, trained readout, and positional baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from train_jcommonsenseqa_ruri_readout import _metrics


class Scorer(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _predict(model: nn.Module, features: list[Tensor]) -> list[Tensor]:
    with torch.inference_mode():
        return list(model(torch.cat(features)).reshape(len(features), 5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    data = torch.load(args.features, map_location="cpu", weights_only=True)
    train_features, validation_features = data["train"], data["validation"]
    train_rows, validation_rows = _rows(args.train), _rows(args.validation)
    targets_train = [int(row["label"]) for row in train_rows]
    targets_validation = [int(row["label"]) for row in validation_rows]
    model = Scorer(int(train_features[0].shape[-1]))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    train_logits = _predict(model, train_features)
    validation_logits = _predict(model, validation_features)
    train_majority = max(set(targets_train), key=targets_train.count)
    chance_logits = [torch.zeros(5) for _ in validation_rows]
    majority_logits = [
        torch.nn.functional.one_hot(torch.tensor(train_majority), 5).float() * 20
        for _ in validation_rows
    ]
    cosine_logits = []
    hidden_size = train_features[0].shape[-1] // 4
    for feature in validation_features:
        query = feature[0, :hidden_size]
        options = feature[:, hidden_size : hidden_size * 2]
        cosine_logits.append(options @ query)
    uniform_metrics = _metrics(chance_logits, targets_validation)
    uniform_metrics["accuracy"] = 0.2
    uniform_metrics["ece_15_bins"] = 0.0
    saved_predictions = _rows(args.predictions)
    replay_probabilities = [torch.softmax(row.float(), dim=-1) for row in validation_logits]
    maximum_probability_delta = max(
        abs(float(value) - float(saved))
        for probabilities, saved in zip(replay_probabilities, saved_predictions, strict=True)
        for value, saved in zip(probabilities, saved["probabilities"], strict=True)
    )
    replay_argmax_matches = sum(
        int(int(logits.argmax()) == int(row["prediction_index"]))
        for logits, row in zip(validation_logits, saved_predictions, strict=True)
    )
    result = {
        "training_metrics_selected_readout": _metrics(train_logits, targets_train),
        "validation_metrics_selected_readout": _metrics(validation_logits, targets_validation),
        "validation_metrics_frozen_ruri_cosine": _metrics(cosine_logits, targets_validation),
        "validation_metrics_uniform_random_choice_expected": uniform_metrics,
        "validation_metrics_train_majority_position": _metrics(majority_logits, targets_validation),
        "train_majority_position": train_majority,
        "selected_checkpoint_replay": {
            "matching_argmax_rows": replay_argmax_matches,
            "total_rows": len(validation_rows),
            "maximum_probability_absolute_delta": maximum_probability_delta,
        },
        "note": (
            "All validation metrics use the same grouped development rows; "
            "no official test exists in the pinned release."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
