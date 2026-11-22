import numpy as np

from scripts.diagnose_native_numeric_span_decoder import (
    deranged_prediction_control,
    fold_for_group,
    numeric_span_features,
    ridge_oof_predictions,
)


def test_numeric_span_features_aligns_values_and_pools_subtokens():
    state = '[{"cost":10,"id":"A","risk":20,"time":30}]'
    prefix = "State: "
    text = prefix + state
    offsets = [(index, index + 1) for index in range(len(text))]
    hidden = np.zeros((len(offsets) + 2, 384), dtype=np.float32)
    for index in range(len(offsets)):
        hidden[index + 1, :] = index

    features = numeric_span_features(state, offsets, hidden)

    assert [(row["field"], row["value"]) for row in features] == [
        ("cost", 10),
        ("risk", 20),
        ("time", 30),
    ]
    assert all(row["feature"].shape == (384,) for row in features)
    assert features[0]["token_piece_count"] == 2


def test_numeric_group_cv_is_deterministic_and_keeps_groups_intact():
    examples = []
    for group_index in range(25):
        group = f"state-{group_index}"
        for item_index in range(4):
            for field_index, field in enumerate(("cost", "risk", "time")):
                value = 10 * group_index + item_index + field_index
                feature = np.zeros(384, dtype=np.float32)
                feature[0] = value
                feature[1] = group_index
                examples.append(
                    {
                        "group": group,
                        "item_id": f"{group}-{item_index}",
                        "field": field,
                        "value": value,
                        "feature": feature,
                    }
                )

    predictions, assignments = ridge_oof_predictions(examples)

    assert len(predictions) == len(examples)
    assert set(assignments.values()) == set(range(5))
    assert assignments == {group: fold_for_group(group) for group in sorted(assignments)}
    assert all(np.isfinite(value) for value in predictions.values())


def test_within_state_shuffle_is_a_complete_derangement():
    predictions = {
        ("g", f"item-{item}", field): float(item)
        for item in range(4)
        for field in ("cost", "risk", "time")
    }

    control = deranged_prediction_control(
        predictions, groups={"g": [f"item-{item}" for item in range(4)]}
    )

    assert len(control) == 12
    for field in ("cost", "risk", "time"):
        assert sorted(control[("g", f"item-{item}", field)] for item in range(4)) == [0, 1, 2, 3]
        assert all(control[("g", f"item-{item}", field)] != item for item in range(4))
