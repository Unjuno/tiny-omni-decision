import pytest

from tiny_omni_decision.factorial_interventions import evaluate_intervention_factorial


def inputs():
    return dict(
        evaluation_role="development",
        candidate_frozen=False,
        sample_ids=["q1", "q2", "q3", "q4"],
        group_ids=["scene-a", "scene-a", "scene-b", "scene-b"],
        sample_ids_by_arm={
            "matched": ["q1", "q2", "q3", "q4"],
            "shuffle_audio": ["q1", "q2", "q3", "q4"],
        },
        group_ids_by_arm={
            "matched": ["scene-a", "scene-a", "scene-b", "scene-b"],
            "shuffle_audio": ["scene-a", "scene-a", "scene-b", "scene-b"],
        },
        option_ids=[["left", "right"]] * 4,
        option_ids_by_arm={
            "matched": [["left", "right"]] * 4,
            "shuffle_audio": [["left", "right"]] * 4,
        },
        intervention_types={"matched": "matched", "shuffle_audio": "shuffle"},
        targets=[0, 1, 0, 1],
        targets_by_arm={
            "matched": [0, 1, 0, 1],
            "shuffle_audio": [0, 1, 0, 1],
        },
        probabilities_by_arm={
            "matched": [[.9, .1], [.2, .8], [.8, .2], [.3, .7]],
            "shuffle_audio": [[.6, .4], [.2, .8], [.4, .6], [.3, .7]],
        },
    )


def test_intervention_comparison_is_paired_and_grouped():
    report = evaluate_intervention_factorial(**inputs())
    assert report["sample_count"] == 4
    assert report["group_count"] == 2
    assert report["metrics"]["matched"]["accuracy"] == 1.0
    comparison = report["paired_group_comparisons"]["shuffle_audio"]
    assert comparison["accuracy_delta"]["mean_difference"] == -0.25
    assert comparison["accuracy_delta"]["clusters"] == 2
    assert report["sample_ids_sha256"] != report["group_ids_sha256"]
    assert report["schema_version"] == 2
    assert report["sample_ids_sha256_by_arm"]["matched"] == report["sample_ids_sha256"]
    assert report["group_ids_sha256_by_arm"]["shuffle_audio"] == report["group_ids_sha256"]
    assert report["targets_sha256_by_arm"]["matched"] == report["targets_sha256"]


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"sample_ids": ["q1", "q1", "q3", "q4"]}, "unique"),
        ({"group_ids": ["scene-a", "scene-a", "", "scene-b"]}, "group"),
        ({"sample_ids_by_arm": {
            "matched": ["q1", "q2", "q3", "q4"],
            "shuffle_audio": ["q2", "q1", "q3", "q4"],
        }}, "sample IDs/order"),
        ({"group_ids_by_arm": {
            "matched": ["scene-a", "scene-a", "scene-b", "scene-b"],
            "shuffle_audio": ["scene-b", "scene-a", "scene-a", "scene-b"],
        }}, "group IDs/order"),
        ({"targets_by_arm": {
            "matched": [0, 1, 0, 1],
            "shuffle_audio": [1, 0, 0, 1],
        }}, "targets/order"),
        ({"sample_ids_by_arm": {"matched": ["q1", "q2", "q3", "q4"]}}, "each prediction arm"),
        ({"intervention_types": {"matched": "matched", "shuffle_audio": "mystery"}},
         "intervention type"),
        ({"option_ids_by_arm": {
            "matched": [["left", "right"]] * 4,
            "shuffle_audio": [["right", "left"], ["left", "right"],
                              ["left", "right"], ["left", "right"]],
        }}, "option"),
        ({
            "probabilities_by_arm": {
                "matched": [[.9, .1]] * 4,
                "bad": [[.8, .3]] * 4,
            },
            "option_ids_by_arm": {
                "matched": [["left", "right"]] * 4,
                "bad": [["left", "right"]] * 4,
            },
            "intervention_types": {"matched": "matched", "bad": "shuffle"},
        }, "normalized"),
    ],
)
def test_invalid_factorial_alignment_fails_closed(updates, message):
    args = inputs()
    args.update(updates)
    with pytest.raises(ValueError, match=message):
        evaluate_intervention_factorial(**args)


def test_sealed_audit_requires_frozen_candidate():
    args = inputs()
    args["evaluation_role"] = "sealed_audit"
    with pytest.raises(ValueError, match="frozen"):
        evaluate_intervention_factorial(**args)
    args["candidate_frozen"] = True
    args["frozen_candidate_sha256"] = "a" * 64
    assert evaluate_intervention_factorial(**args)["evaluation_role"] == "sealed_audit"
