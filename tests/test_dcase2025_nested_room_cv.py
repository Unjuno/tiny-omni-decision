from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.run_dcase2025_nested_room_cv import fit_fold_mode, room_folds


def test_room_cv_keeps_room_groups_disjoint_and_tests_each_row_once():
    rows = [
        {
            "sample_id": f"{room}-{index}",
            "parent_recording": f"{room}-parent-{index // 2}",
            "room_id": room,
        }
        for room in ("r1", "r2", "r3", "r4")
        for index in range(3)
    ]

    folds = room_folds(rows, seed=17)

    assert len(folds) == 4
    test_counts = np.zeros(len(rows), dtype=np.int64)
    for fold in folds:
        train_rooms = set(fold["train_rooms"])
        assert fold["test_room"] not in train_rooms
        assert fold["validation_room"] not in train_rooms
        assert fold["test_room"] != fold["validation_room"]
        test_counts[fold["test_indices"]] += 1
    assert (test_counts == 1).all()


def test_room_cv_is_deterministic_and_rejects_too_few_rooms():
    rows = [
        {"sample_id": room, "parent_recording": room, "room_id": room}
        for room in ("r1", "r2", "r3")
    ]

    first = room_folds(rows, seed=17)
    second = room_folds(rows, seed=17)

    assert [(fold["test_room"], fold["validation_room"]) for fold in first] == [
        (fold["test_room"], fold["validation_room"]) for fold in second
    ]
    with pytest.raises(ValueError, match="at least three"):
        room_folds(rows[:2], seed=17)


def test_room_cv_rejects_parent_recording_overlap():
    rows = [
        {"sample_id": room, "parent_recording": "shared-parent", "room_id": room}
        for room in ("r1", "r2", "r3")
    ]

    with pytest.raises(ValueError, match="parent recording overlaps"):
        room_folds(rows, seed=17)


def test_readout_fit_selects_from_inner_validation_only():
    rng = np.random.default_rng(17)
    train_audio = rng.normal(size=(24, 768)).astype(np.float32)
    train_video = rng.normal(size=(24, 768)).astype(np.float32)
    validation_audio = rng.normal(size=(12, 768)).astype(np.float32)
    validation_video = rng.normal(size=(12, 768)).astype(np.float32)
    train_events = np.arange(24, dtype=np.int64) % 13
    validation_events = np.arange(12, dtype=np.int64) % 13
    train_targets = np.arange(24, dtype=np.int64) % 2
    validation_targets = np.arange(12, dtype=np.int64) % 2

    _model, fit = fit_fold_mode(
        "joint",
        train_audio,
        train_video,
        train_events,
        train_targets,
        validation_audio,
        validation_video,
        validation_events,
        validation_targets,
        seed=17,
        max_epochs=3,
        patience=2,
        batch_size=8,
    )

    assert 1 <= fit["best_epoch"] <= fit["epochs_run"] <= 3
    assert np.isfinite(fit["best_validation_nll"])
    assert fit["selected_validation_metrics"]["count"] == 12
