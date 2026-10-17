import pytest
import torch

from scripts.train_clevrer_unique_coverage import (
    _bootstrap_difference,
    _group_metrics,
    _sample_batches,
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
    validation = _scene(5, split="validation") + _scene(6, split="validation")

    report = _validate_row_groups(baseline, additional, validation)

    assert report["baseline"]["scenes"] == 2
    assert report["additional"]["scenes"] == 2
    assert report["candidate"]["scenes"] == 4
    assert report["validation"]["scenes"] == 2
    assert report["baseline_additional_scene_overlap"] == 0
    assert report["train_validation_media_overlap"] == 0


def test_group_accounting_rejects_validation_scene_reuse() -> None:
    baseline = _scene(1)
    additional = _scene(2)
    validation = _scene(2, split="validation")

    with pytest.raises(ValueError, match="scene identities overlap"):
        _validate_row_groups(baseline, additional, validation)


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
