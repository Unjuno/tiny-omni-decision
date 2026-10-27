from scripts.evaluate_grid_direct_speaker_pilot import metrics, speaker_bootstrap


def rows_fixture():
    return [
        {
            "speaker": "s1",
            "target_id": "a",
            "option_ids": ["a", "b", "c", "d"],
            "joint": {
                "selected_option_id": "a",
                "probabilities": {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1},
            },
            "control": {
                "selected_option_id": "b",
                "probabilities": {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            },
        },
        {
            "speaker": "s2",
            "target_id": "c",
            "option_ids": ["a", "b", "c", "d"],
            "joint": {
                "selected_option_id": "d",
                "probabilities": {"a": 0.1, "b": 0.1, "c": 0.2, "d": 0.6},
            },
            "control": {
                "selected_option_id": "a",
                "probabilities": {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            },
        },
    ]


def test_metrics_reports_multiclass_accuracy_nll_brier_and_ece():
    result = metrics(rows_fixture(), "joint")

    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["nll"] > 0
    assert result["brier"] > 0
    assert result["ece_10bin"] > 0


def test_speaker_bootstrap_is_deterministic_and_paired():
    result = speaker_bootstrap(rows_fixture(), "joint", "control", iterations=100)

    assert result["group"] == "speaker"
    assert result["speakers"] == ["s1", "s2"]
    assert result["mean_accuracy_difference"] == 0.5
    assert result["percentile_95_interval"][0] <= 0.5
    assert result["percentile_95_interval"][1] >= 0.5
