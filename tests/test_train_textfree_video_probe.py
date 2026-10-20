import pytest

torch = pytest.importorskip("torch")


def test_scene_shuffle_changes_each_scene_and_keeps_repeated_rows_together():
    from scripts.train_textfree_video_probe import _scene_shuffled_video
    from tiny_omni_decision.textfree_video import EncodedVideoRows

    source = torch.stack(
        [
            torch.full((8, 768), 1.0),
            torch.full((8, 768), 1.0),
            torch.full((8, 768), 2.0),
            torch.full((8, 768), 2.0),
            torch.full((8, 768), 3.0),
            torch.full((8, 768), 3.0),
        ]
    )
    batch = EncodedVideoRows(
        video=source,
        program_ids=torch.ones((6, 1), dtype=torch.long),
        taxonomy_ids=torch.ones(6, dtype=torch.long),
        option_ids=torch.ones((6, 2), dtype=torch.long),
        option_mask=torch.ones((6, 2), dtype=torch.bool),
        targets=torch.zeros(6, dtype=torch.long),
        sample_ids=tuple(str(index) for index in range(6)),
    )
    rows = [{"scene_group_id": scene} for scene in ("a", "a", "b", "b", "c", "c")]

    shuffled = _scene_shuffled_video(batch, rows)

    assert all(not torch.equal(source[index], shuffled[index]) for index in range(6))
    assert torch.equal(shuffled[0], shuffled[1])
    assert torch.equal(shuffled[2], shuffled[3])
    assert torch.equal(shuffled[4], shuffled[5])


def test_scene_bootstrap_uses_paired_scene_units_and_fixed_seed():
    from scripts.train_textfree_video_probe import _paired_scene_bootstrap

    left = {
        f"row-{scene}": {
            "scene_group_id": f"scene-{scene}",
            "question_type": "temporal_descriptive",
            "accuracy": 1.0,
            "nll": 0.5,
            "brier": 0.25,
        }
        for scene in range(4)
    }
    right = {
        f"row-{scene}": {
            "scene_group_id": f"scene-{scene}",
            "question_type": "temporal_descriptive",
            "accuracy": 0.0,
            "nll": 1.5,
            "brier": 1.25,
        }
        for scene in range(4)
    }

    result = _paired_scene_bootstrap(
        left, right, "temporal_descriptive", "accuracy", "left", "right"
    )

    assert result["scene_count"] == 4
    assert result["difference"] == pytest.approx(1.0)
    assert result["percentile_95_ci"] == pytest.approx([1.0, 1.0])
    assert result["bootstrap_seed"] == 17


@pytest.mark.parametrize(
    ("train_count", "epochs", "validation_interval_epochs"),
    [(1024, 128, 2), (2048, 64, 1)],
)
def test_coverage_schedules_match_updates_and_validation_opportunities(
    train_count, epochs, validation_interval_epochs
):
    from scripts.train_textfree_video_probe import (
        _training_schedule,
        _validation_interval_updates,
    )

    schedule = _training_schedule(
        train_count=train_count,
        batch_size=64,
        epochs=epochs,
        max_optimizer_updates=2048,
    )
    validation_updates = _validation_interval_updates(
        train_count=train_count,
        batch_size=64,
        validation_interval_epochs=validation_interval_epochs,
        epochs=epochs,
    )

    assert schedule["total_optimizer_updates"] == 2048
    assert schedule["full_run_example_exposures"] == 131072
    assert validation_updates == 32


def test_training_schedule_rejects_budget_that_does_not_match_epoch_order():
    from scripts.train_textfree_video_probe import _training_schedule

    with pytest.raises(ValueError, match="do not match"):
        _training_schedule(
            train_count=2048,
            batch_size=64,
            epochs=128,
            max_optimizer_updates=2048,
        )


def test_media_integrity_receipt_accepts_frozen_fetch_report_schema():
    from scripts.train_textfree_video_probe import _validate_media_integrity_report

    source = {
        "train_manifest_sha256": "train-hash",
        "validation_manifest_sha256": "validation-hash",
    }
    receipt = {
        "status": "passed",
        "all_media_sha256_verified": True,
        "sample_hashes": {
            "train_jsonl": "train-hash",
            "validation_jsonl": "validation-hash",
        },
        "group_integrity": {
            "train_validation_scene_overlap": 0,
            "train_validation_media_overlap": 0,
        },
    }

    _validate_media_integrity_report(receipt, source)

    receipt["sample_hashes"]["validation_jsonl"] = "wrong-hash"
    with pytest.raises(ValueError, match="integrity report"):
        _validate_media_integrity_report(receipt, source)


def test_prediction_receipt_uses_typed_json_and_normalized_option_probabilities(tmp_path):
    import json

    from scripts.train_textfree_video_probe import _write_predictions
    from tiny_omni_decision.textfree_video import build_video_vocab, encode_video_rows

    row = {
        "id": "scene-a-q1",
        "media_sha256": "a" * 64,
        "scene_group_id": "scene-a",
        "question": "not passed to the scorer",
        "program": ["events", "count"],
        "taxonomy": "count",
        "question_type": "temporal_descriptive",
        "options": ["0", "1", "2"],
        "target": "1",
    }
    vocab = build_video_vocab([row])
    batch = encode_video_rows([row], {"a" * 64: np_zero_video()}, vocab)
    destination = tmp_path / "predictions.jsonl"

    _write_predictions(
        destination,
        torch.tensor([[0.0, 3.0, -1.0]]),
        batch,
        [row],
        vocab.option_to_id,
    )

    prediction = json.loads(destination.read_text(encoding="utf-8"))
    assert prediction["schema_version"] == "1.0"
    assert prediction["status"] == "selected"
    assert prediction["selected_option_id"] == "1"
    assert [option["option_id"] for option in prediction["options"]] == ["0", "1", "2"]
    assert sum(option["probability"] for option in prediction["options"]) == pytest.approx(1.0)


def np_zero_video():
    return torch.zeros((8, 768), dtype=torch.float32)
