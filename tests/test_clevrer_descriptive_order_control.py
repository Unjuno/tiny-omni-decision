from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("transformers")
control = importlib.import_module("scripts.evaluate_clevrer_descriptive_order_control")


def _sample() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw = []
    media = []
    for scene in range(200):
        filename = f"video_{scene:05d}.mp4"
        raw.append(
            {
                "scene_index": scene,
                "video_filename": filename,
                "questions": [
                    {
                        "question_id": 0,
                        "question_type": "descriptive",
                        "question_subtype": "exist",
                        "question": f"Is a cube visible in scene {scene}?",
                        "answer": "yes",
                        "program": ["objects", "exist"],
                    },
                    {
                        "question_id": 1,
                        "question_type": "descriptive",
                        "question_subtype": "exist",
                        "question": f"Is a cube there at the end in scene {scene}?",
                        "answer": "no",
                        "program": ["objects", "end", "get_frame", "exist"],
                    },
                ],
            }
        )
        media.append(
            {
                "scene_index": scene,
                "video_filename": filename,
                "split": "validation",
                "media_sha256": f"{scene:064x}",
                "media_path": f"media/validation/{filename}",
            }
        )
    return control.expand_descriptive_questions(raw, media), raw


def test_source_audit_confirms_temporal_static_pairs_and_upstream_labels() -> None:
    rows, raw = _sample()

    report = control._validate_source_rows(rows, raw)

    assert report["sample_rows"] == 400
    assert report["scene_count"] == 200
    assert report["question_type_counts"] == {
        "static_descriptive": 200,
        "temporal_descriptive": 200,
    }
    assert report["source_answer_mapping_matches"] == 400


def test_source_audit_rejects_changed_target() -> None:
    rows, raw = _sample()
    rows[0]["target"] = "no"

    with pytest.raises(ValueError, match="source CLEVRER answer mapping"):
        control._validate_source_rows(rows, raw)


def test_paired_scene_bootstrap_is_zero_for_identical_predictions() -> None:
    rows, _ = _sample()
    probabilities = [[0.2, 0.8] if row["target"] == "yes" else [0.8, 0.2] for row in rows]
    preds = [
        {
            "scene_index": row["scene_index"],
            "question_type": row["question_type"],
            "target_index": list(row["options"]).index(row["target"]),
            "probabilities": probability,
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]

    result = control._paired_scene_bootstrap(
        preds, preds, rows, replicates=100, seed=17
    )

    for metrics in result["delta_intervals"].values():
        for bounds in metrics.values():
            assert bounds == [0.0, 0.0]


def test_probability_metrics_cover_calibration_and_multiclass_targets() -> None:
    rows = [
        {"target_index": 1, "probabilities": [0.25, 0.75]},
        {"target_index": 2, "probabilities": [0.1, 0.2, 0.7]},
    ]

    metrics = control._metrics(rows)

    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] > 0
    assert 0 <= metrics["ece_15_bins"] <= 1
