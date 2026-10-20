from __future__ import annotations

import json

from scripts.train_clevrer_temporal_coverage_probe import (
    _prediction_map,
    _scene_bootstrap_difference,
)


def test_prediction_map_reads_serialized_classification_options(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "scene_group_id": "scene-1",
                "target": "yes",
                "prediction": "no",
                "options": [
                    {"option_id": "no", "probability": 0.25},
                    {"option_id": "yes", "probability": 0.75},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _prediction_map(path)["sample-1"]

    assert result["scene_group_id"] == "scene-1"
    assert result["accuracy"] == 0.0
    assert result["nll"] > 0.28
    assert result["brier"] == 0.125


def test_scene_bootstrap_pairs_by_scene_and_is_seeded() -> None:
    left = {
        f"sample-{i}": {
            "scene_group_id": f"scene-{i // 2}",
            "accuracy": 1.0,
            "nll": 0.4,
            "brier": 0.2,
        }
        for i in range(8)
    }
    right = {
        sample_id: {
            "scene_group_id": row["scene_group_id"],
            "accuracy": 0.5,
            "nll": 0.6,
            "brier": 0.3,
        }
        for sample_id, row in left.items()
    }

    result = _scene_bootstrap_difference(left, right, "left", "right", "accuracy")
    repeated = _scene_bootstrap_difference(left, right, "left", "right", "accuracy")

    assert result == repeated
    assert result["difference"] == 0.5
    assert result["scenes"] == 4
