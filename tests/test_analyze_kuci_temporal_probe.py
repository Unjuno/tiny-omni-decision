import pytest

from scripts.analyze_kuci_temporal_probe import paired_group_bootstrap


def _rows() -> list[dict[str, object]]:
    base = {
        "choice_a": "選択肢A",
        "choice_b": "選択肢B",
        "choice_c": "選択肢C",
        "choice_d": "選択肢D",
        "agreement": 3,
    }
    return [
        base | {"id": "1", "label": "a", "core_event_pair": "pair-1", "context": "context 1"},
        base | {"id": "2", "label": "b", "core_event_pair": "pair-1", "context": "context 2"},
        base | {"id": "3", "label": "c", "core_event_pair": "pair-3", "context": "shared"},
        base | {"id": "4", "label": "d", "core_event_pair": "pair-4", "context": "shared"},
    ]


def _predictions(rows: list[dict[str, object]], predicted: list[str]) -> list[dict[str, object]]:
    output = []
    for row, answer in zip(rows, predicted, strict=True):
        confidence = 0.7
        probabilities = [0.1] * 4
        probabilities["abcd".index(answer)] = confidence
        remainder = (1 - confidence) / 3
        probabilities = [
            confidence if item == "abcd".index(answer) else remainder for item in range(4)
        ]
        output.append(
            {
                "id": row["id"],
                "target": row["label"],
                "prediction": answer,
                "probabilities_abcd": probabilities,
            }
        )
    return output


def test_bootstrap_pairs_predictions_by_id_and_resamples_connected_groups() -> None:
    rows = _rows()
    baseline = _predictions(rows, ["a", "a", "c", "c"])
    candidate = _predictions(rows, ["a", "b", "c", "c"])

    result = paired_group_bootstrap(rows, baseline, candidate, replicates=100, seed=17)

    assert result["connected_groups"] == 2
    assert result["metrics"]["accuracy"]["candidate_minus_baseline"] == 0.25
    assert result["metrics"]["nll"]["candidate_minus_baseline"] < 0


def test_bootstrap_rejects_mismatched_prediction_order() -> None:
    rows = _rows()
    predictions = _predictions(rows, ["a", "b", "c", "d"])

    with pytest.raises(ValueError, match="IDs/order"):
        paired_group_bootstrap(rows, predictions, list(reversed(predictions)), replicates=10)
