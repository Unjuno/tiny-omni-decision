import numpy as np

from scripts.pch_moment_sensor_probe import (
    assign_subject_splits,
    canonical_activity,
    participant_bootstrap_comparison,
    resample_and_window,
    stratify_validation_with_fit_only_labels,
    subject_id_from_member,
    timestamp_segments,
)


def test_subject_id_accepts_pch_filename_variants():
    assert subject_id_from_member("P008 - 3 - WashDish.csv") == "P008"
    assert subject_id_from_member("P099chopping.csv") == "P099"


def test_hash_split_is_deterministic_and_grouped():
    names = [f"P{index:03d}_{index}_walking.csv" for index in range(1, 31)]

    first, first_hashes = assign_subject_splits(names)
    second, second_hashes = assign_subject_splits(names)

    assert first == second
    assert first_hashes == second_hashes
    assert set(first.values()) == {"fit", "validation", "heldout"}


def test_validation_coverage_swaps_fit_subjects_without_parsing_heldout(monkeypatch):
    import scripts.pch_moment_sensor_probe as probe

    monkeypatch.setattr(probe, "ACTIVITY_IDS", ("walking", "jogging"))
    names = [
        "P001_jogging.csv",
        "P002_jogging.csv",
        "P003_catch.csv",
        "P006_catch.csv",
        "P004_walking.csv",
        "P005_walking.csv",
        "P999_unmapped-heldout-label.csv",
    ]
    splits = {
        "P001": "validation",
        "P002": "validation",
        "P003": "validation",
        "P006": "validation",
        "P004": "fit",
        "P005": "fit",
        "P999": "heldout",
    }

    stratified = stratify_validation_with_fit_only_labels(names, splits)

    assert stratified["P999"] == "heldout"
    assert sum(value == "validation" for value in stratified.values()) == 4
    validation_counts = {
        activity: {
            subject_id_from_member(name)
            for name in names
            if stratified[subject_id_from_member(name)] == "validation"
            and canonical_activity(name) == activity
        }
        for activity in ("walking", "jogging")
    }
    assert all(len(subjects) >= 2 for subjects in validation_counts.values())


def test_resampling_produces_finite_nonoverlapping_triaxial_windows():
    sample_rate = 51.2
    timestamps = np.arange(256) * (1000 / sample_rate)
    axes = np.stack(
        (np.sin(np.arange(256) / 10), np.cos(np.arange(256) / 10), np.ones(256)),
        axis=1,
    ).astype(np.float32)

    windows = resample_and_window(timestamps, axes)

    assert windows.shape == (2, 3, 128)
    assert np.isfinite(windows).all()


def test_resampling_splits_clock_resets_and_large_gaps():
    sample_interval = 1000 / 51.2
    timestamps = np.concatenate(
        (np.arange(128) * sample_interval, np.arange(128) * sample_interval)
    )
    axes = np.ones((256, 3), dtype=np.float32)

    assert len(timestamp_segments(timestamps)) == 2
    windows = resample_and_window(timestamps, axes)
    assert windows.shape == (2, 3, 128)


def test_participant_bootstrap_is_deterministic_and_paired():
    target = np.asarray([0, 1, 0, 1])
    subjects = ["P001", "P001", "P002", "P002"]
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]], dtype=np.float64)

    result = participant_bootstrap_comparison(
        probabilities, probabilities, target, subjects, replicates=200
    )

    assert result["delta_accuracy_95_percent_interval"] == [0.0, 0.0]
    assert result["delta_nll_95_percent_interval"] == [0.0, 0.0]
