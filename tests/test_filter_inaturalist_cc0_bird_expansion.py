import json

import pytest

from scripts.filter_inaturalist_cc0_bird_expansion import filter_expansion, freeze_expansion


def _row(split, observer, observation, photo, taxon=1):
    return {
        "split": split,
        "observer_group_sha256": observer,
        "observation_id": observation,
        "photo_id": photo,
        "taxon_id": taxon,
        "observation_license": "cc0",
        "photo_license": "cc0",
    }


def test_expansion_excludes_previous_observers_observations_and_photos():
    previous = [_row("train", "old-observer", "old-observation", "old-photo")]
    candidate = {
        "train": [
            _row("train", "old-observer", "new-observation", "new-photo"),
            _row("train", "new-observer", "old-observation", "new-photo-2"),
            _row("train", "new-observer-2", "new-observation-2", "old-photo"),
            _row("train", "fresh-observer", "fresh-observation", "fresh-photo"),
        ],
        "validation": [
            _row("validation", "validation-observer", "validation-observation", "validation-photo")
        ],
    }

    retained, summary = filter_expansion(previous, candidate)

    assert [row["observation_id"] for row in retained["train"]] == ["fresh-observation"]
    assert [row["observation_id"] for row in retained["validation"]] == [
        "validation-observation"
    ]
    assert summary["excluded_previous_identity_counts"] == {
        "previous_observer": 1,
        "previous_observation": 1,
        "previous_photo": 1,
    }
    assert summary["identity_disjoint_from_previous"] is True
    assert summary["train_validation_identity_disjoint"] is True
    assert summary["eligible_for_training"] is False


def test_expansion_fails_closed_on_wrong_split_or_non_cc0_row():
    previous = [_row("train", "old", "old-observation", "old-photo")]
    wrong_split = {
        "train": [_row("validation", "fresh", "new-observation", "new-photo")],
        "validation": [_row("validation", "fresh-val", "v-observation", "v-photo")],
    }
    with pytest.raises(ValueError, match="wrong split"):
        filter_expansion(previous, wrong_split)

    denied = {
        "train": [_row("train", "fresh", "new-observation", "new-photo")],
        "validation": [
            _row("validation", "fresh-val", "v-observation", "v-photo")
            | {"photo_license": "cc-by-nc"}
        ],
    }
    with pytest.raises(ValueError, match="explicitly CC0"):
        filter_expansion(previous, denied)


def test_expansion_rejects_duplicate_observer_class_and_records_missing_class_gate():
    previous = [_row("train", "old", "old-observation", "old-photo", taxon=1)]
    duplicate = {
        "train": [
            _row("train", "fresh", "new-1", "photo-1"),
            _row("train", "fresh", "new-2", "photo-2"),
        ],
        "validation": [_row("validation", "fresh-val", "v-1", "v-photo")],
    }
    with pytest.raises(ValueError, match="duplicate observer"):
        filter_expansion(previous, duplicate)

    missing_class = {
        "train": [_row("train", "fresh", "new", "photo", taxon=2)],
        "validation": [_row("validation", "fresh-val", "v", "v-photo", taxon=2)],
    }
    retained, summary = filter_expansion(previous, missing_class)
    assert len(retained["train"]) == 1
    assert summary["all_classes_represented"] is False
    assert summary["minimum_count_gate"]["passed"] is False
    assert summary["eligible_for_training"] is False


def test_freeze_expansion_applies_sources_sequentially_and_keeps_provenance(tmp_path):
    previous = tmp_path / "previous"
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    output = tmp_path / "output"
    for directory in (previous, source_a, source_b):
        directory.mkdir()
        (directory / "metadata.json").write_text("{}\n", encoding="utf-8")
        (directory / "train.jsonl").write_text("", encoding="utf-8")
        (directory / "validation.jsonl").write_text("", encoding="utf-8")
    (source_a / "train.jsonl").write_text(
        json.dumps(_row("train", "observer-a", "obs-a", "photo-a")) + "\n",
        encoding="utf-8",
    )
    (source_a / "validation.jsonl").write_text(
        json.dumps(_row("validation", "observer-b", "obs-b", "photo-b")) + "\n",
        encoding="utf-8",
    )
    (source_b / "train.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _row("train", "observer-a", "obs-a-duplicate", "photo-a-duplicate"),
                _row("train", "observer-c", "obs-c", "photo-c"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = freeze_expansion([previous], [source_a, source_b], output)

    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert {row["observation_id"] for row in train_rows} == {"obs-a", "obs-c"}
    assert result["summary"]["new_train_count"] == 2
    assert result["summary"]["excluded_previous_identity_counts"] == {
        "previous_observer": 1
    }
    assert len(result["expanded_source_sha256"]) == 2
