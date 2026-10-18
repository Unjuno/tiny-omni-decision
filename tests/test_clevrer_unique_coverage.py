import pytest
import torch

from scripts.train_clevrer_unique_coverage import (
    _bootstrap_difference,
    _group_metrics,
    _sample_batches,
    _sampler_audit,
    _validate_previous_scene_exclusions,
    _validate_row_groups,
)


def _row(scene: int, kind: str, *, split: str = "train") -> dict:
    return {
        "id": f"{split}-{scene}-{kind}",
        "scene_index": scene,
        "media_sha256": f"media-{scene}",
        "question_type": kind,
        "taxonomy": "count",
        "question": f"question {kind}",
        "options": ["0", "1"],
        "target": "1",
    }


def _scene(scene: int, *, split: str = "train") -> list[dict]:
    return [
        _row(scene, "static_descriptive", split=split),
        _row(scene, "temporal_descriptive", split=split),
    ]


def test_group_accounting_keeps_scenes_and_media_disjoint() -> None:
    baseline = _scene(1) + _scene(2)
    additional = _scene(3) + _scene(4)
    baseline_validation = _scene(7, split="validation") + _scene(8, split="validation")
    validation = _scene(5, split="validation") + _scene(6, split="validation")

    report = _validate_row_groups(baseline, additional, baseline_validation, validation)

    assert report["baseline"]["scenes"] == 2
    assert report["additional"]["scenes"] == 2
    assert report["candidate"]["scenes"] == 4
    assert report["baseline_validation"]["scenes"] == 2
    assert report["validation"]["scenes"] == 2
    assert report["baseline_additional_scene_overlap"] == 0
    assert report["train_validation_media_overlap"] == 0


def test_paired_bootstrap_supports_variable_option_counts() -> None:
    baseline = [
        {
            "id": "q1", "scene_index": 1, "question_type": "static_descriptive",
            "options": ["0", "1"], "target": "1", "prediction": "1",
            "probabilities": [0.2, 0.8],
        },
        {
            "id": "q2", "scene_index": 2, "question_type": "temporal_descriptive",
            "options": ["before", "same", "after"], "target": "after",
            "prediction": "same", "probabilities": [0.1, 0.6, 0.3],
        },
        {
            "id": "q3", "scene_index": 1, "question_type": "temporal_descriptive",
            "options": ["before", "same", "after"], "target": "before",
            "prediction": "before", "probabilities": [0.7, 0.2, 0.1],
        },
        {
            "id": "q4", "scene_index": 2, "question_type": "static_descriptive",
            "options": ["0", "1"], "target": "0", "prediction": "0",
            "probabilities": [0.9, 0.1],
        },
    ]
    candidate = [
        {**baseline[0], "prediction": "0", "probabilities": [0.6, 0.4]},
        {**baseline[1], "prediction": "after", "probabilities": [0.1, 0.2, 0.7]},
        {**baseline[2], "prediction": "before", "probabilities": [0.8, 0.1, 0.1]},
        {**baseline[3], "prediction": "1", "probabilities": [0.3, 0.7]},
    ]

    result = _bootstrap_difference(baseline, candidate, resamples=20, seed=17)

    assert result["groups"] == 2
    assert result["candidate_minus_baseline"]["all"]["brier"]["point"] == pytest.approx(0.2)


def test_group_accounting_rejects_validation_scene_reuse() -> None:
    baseline = _scene(1)
    additional = _scene(2)
    baseline_validation = _scene(7, split="validation")
    validation = _scene(2, split="validation")

    with pytest.raises(ValueError, match="scene identities overlap"):
        _validate_row_groups(baseline, additional, baseline_validation, validation)


def test_previous_scene_gate_requires_baseline_scenes_and_rejects_reuse() -> None:
    baseline_train = _scene(10)
    additional = _scene(11)
    baseline_validation = _scene(10_010, split="validation")
    validation = _scene(10_011, split="validation")
    exclusions = {"train": [10], "validation": [10_010]}

    _validate_previous_scene_exclusions(
        additional, validation, baseline_train, baseline_validation, exclusions
    )

    with pytest.raises(ValueError, match="omitted baseline training"):
        _validate_previous_scene_exclusions(
            additional, validation, baseline_train, baseline_validation,
            {"train": [], "validation": [10_010]},
        )
    with pytest.raises(ValueError, match="previously used scene"):
        _validate_previous_scene_exclusions(
            baseline_train, validation, baseline_train, baseline_validation, exclusions
        )


def test_step_sampler_is_deterministic_and_covers_each_complete_pass() -> None:
    first = list(_sample_batches(sample_count=16, batch_size=8, steps=4, seed=17))
    replay = list(_sample_batches(sample_count=16, batch_size=8, steps=4, seed=17))

    assert first == replay
    for start in (0, 2):
        pass_rows = first[start] + first[start + 1]
        assert sorted(pass_rows) == list(range(16))
        assert len(set(pass_rows)) == 16


def test_step_sampler_rejects_partial_pool_passes() -> None:
    with pytest.raises(ValueError, match="divisible by batch_size"):
        list(_sample_batches(sample_count=10, batch_size=4, steps=1, seed=17))


def test_sampler_audit_counts_presentations_unique_scenes_and_media() -> None:
    rows = _scene(1) + _scene(2) + _scene(3) + _scene(4)

    report = _sampler_audit(rows, batch_size=8, steps=4, seed=17, prefix_presentations=16)

    assert report["total_presentations"] == 32
    assert report["unique_questions_consumed"] == 8
    assert report["repeated_presentations"] == 24
    assert report["unique_scenes_consumed"] == 4
    assert report["unique_media_consumed"] == 4
    assert report["prefix_presentations"] == 16
    assert report["prefix_unique_scenes"] == 4
    assert sum(report["prefix_question_type_counts"].values()) == 16


def test_group_metrics_reports_both_temporal_and_static_tasks() -> None:
    rows = _scene(1) + _scene(2, split="validation")
    targets = [1, 1, 1, 1]
    logits = [torch.tensor([0.0, 1.0]) for _ in rows]

    report = _group_metrics(logits, targets, rows)

    assert report["macro"]["accuracy"] == pytest.approx(1.0)
    assert set(report["by_question_type"]) == {"static_descriptive", "temporal_descriptive"}
    assert report["by_question_type"]["static_descriptive"]["count"] == 2


def test_scene_bootstrap_pairs_same_questions_and_reports_subtasks() -> None:
    rows = []
    for scene in range(8):
        for kind in ("static_descriptive", "temporal_descriptive"):
            options = ["0", "1"]
            rows.append({
                "id": f"validation-{scene}-{kind}",
                "scene_index": scene,
                "question_type": kind,
                "target": "1",
                "prediction": "1",
                "options": options,
                "probabilities": [0.1, 0.9],
            })

    report = _bootstrap_difference(rows, rows, resamples=25, seed=17)

    assert report["groups"] == 8
    assert set(report["candidate_minus_baseline"]) == {
        "all", "static_descriptive", "temporal_descriptive"
    }
    assert report["candidate_minus_baseline"]["all"]["accuracy"]["lower_95"] == 0.0
    assert report["candidate_minus_baseline"]["all"]["nll"]["upper_95"] == 0.0
    assert report["candidate_minus_baseline"]["all"]["brier"]["upper_95"] == 0.0
    assert report["candidate_minus_baseline"]["all"]["ece_15_bins"]["lower_95"] == 0.0


def test_scene_bootstrap_covers_brier_and_ece_differences() -> None:
    rows = []
    for scene in range(4):
        for kind in ("static_descriptive", "temporal_descriptive"):
            rows.append({
                "id": f"validation-{scene}-{kind}",
                "scene_index": scene,
                "question_type": kind,
                "target": "1",
                "prediction": "1",
                "options": ["0", "1"],
                "probabilities": [0.1, 0.9],
            })
    candidate = [{**row, "probabilities": [0.2, 0.8]} for row in rows]

    report = _bootstrap_difference(rows, candidate, resamples=25, seed=17)
    overall = report["candidate_minus_baseline"]["all"]

    assert overall["brier"]["point"] == pytest.approx(0.06)
    assert overall["ece_15_bins"]["point"] == pytest.approx(0.1)
