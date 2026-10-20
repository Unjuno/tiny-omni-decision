import pytest

from scripts.compare_textfree_video_coverage import compare_prediction_sets


def _predictions(probability: float = 0.75):
    rows = []
    for scene in (1, 2):
        for question_type in ("static_descriptive", "temporal_descriptive"):
            rows.append(
                {
                    "id": f"scene-{scene}-{question_type}",
                    "scene_group_id": f"clevrer:validation:{scene}",
                    "question_type": question_type,
                    "target": "yes",
                    "prediction": "yes",
                    "options": [
                        {"option_id": "yes", "probability": probability},
                        {"option_id": "no", "probability": 1.0 - probability},
                    ],
                }
            )
    return {"mean_mlp": rows, "bilstm": [dict(row) for row in rows]}


def test_paired_scene_comparison_reports_zero_difference_for_identical_fits():
    predictions = _predictions()

    report = compare_prediction_sets(
        predictions,
        predictions,
        bootstrap_repetitions=20,
        seed=17,
    )

    assert report["status"] == "PASS_PAIRED_NESTED_COVERAGE_COMPARISON"
    assert report["paired_validation_examples"] == 4
    assert report["paired_validation_scenes"] == 2
    for arm in ("mean_mlp", "bilstm"):
        for metric in ("accuracy", "nll", "brier", "ece_15"):
            result = report["scene_bootstrap"]["large_minus_small"][arm]["macro"][metric]
            assert result["large_minus_small"] == pytest.approx(0.0)
            assert result["percentile_95_ci"] == pytest.approx([0.0, 0.0])


def test_paired_scene_comparison_rejects_mismatched_validation_order():
    small = _predictions()
    large = _predictions()
    large["mean_mlp"] = list(reversed(large["mean_mlp"]))

    with pytest.raises(ValueError, match="same validation ID order"):
        compare_prediction_sets(small, large, bootstrap_repetitions=10)
