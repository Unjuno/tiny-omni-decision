import pytest

from scripts.build_fsd50k_annotation_decision_manifest import (
    ALLOWED_LICENSES,
    TARGET_LABELS,
    build_split_rows,
    summarize_split,
    validate_splits,
)


def test_split_is_deterministic_and_groups_never_cross_splits():
    license_url = sorted(ALLOWED_LICENSES)[0]
    metadata = {
        "a": {"license": license_url, "uploader": "alice", "title": "a"},
        "b": {"license": license_url, "uploader": "alice", "title": "b"},
        "c": {"license": license_url, "uploader": "bob", "title": "c"},
    }
    source = """fname,labels,mids,split
a,Animal,/animal,train
b,Music,/music,train
c,Vehicle,/vehicle,train
"""
    first = build_split_rows(source, metadata, excluded_ids=set(), excluded_uploaders=set())
    second = build_split_rows(source, metadata, excluded_ids=set(), excluded_uploaders=set())
    assert first == second
    uploader_splits = {row["group_id"]: split for split, rows in first.items() for row in rows}
    assert uploader_splits["alice"] in {"train", "calibration", "validation"}
    assert first[uploader_splits["alice"]][0]["sample_id"] == "a"
    assert (
        len({row["split"] for rows in first.values() for row in rows if row["group_id"] == "alice"})
        == 1
    )


def test_filters_noncommercial_and_previously_exposed_uploader_groups():
    allowed = sorted(ALLOWED_LICENSES)[0]
    metadata = {
        "used": {"license": allowed, "uploader": "old-user", "title": "used"},
        "same-user": {"license": allowed, "uploader": "old-user", "title": "same"},
        "noncommercial": {
            "license": "http://creativecommons.org/licenses/by-nc/3.0/",
            "uploader": "new-user",
            "title": "excluded",
        },
        "fresh": {"license": allowed, "uploader": "fresh-user", "title": "fresh"},
    }
    source = (
        "fname,labels,mids,split\n"
        "used,Animal,/animal,train\n"
        "same-user,Music,/music,train\n"
        "noncommercial,Vehicle,/vehicle,train\n"
        "fresh,Human_voice,/voice,train\n"
    )
    result = build_split_rows(
        source, metadata, excluded_ids={"used"}, excluded_uploaders={"old-user"}
    )
    rows = [row for split_rows in result.values() for row in split_rows]
    assert [row["sample_id"] for row in rows] == ["fresh"]


def test_annotation_targets_allow_cooccurrence_and_do_not_redefine_absence_as_truth():
    metadata = {
        "multi": {
            "license": sorted(ALLOWED_LICENSES)[0],
            "uploader": "alice",
            "title": "multi",
        }
    }
    source = "fname,labels,mids,split\nmulti,Animal,/,train\nmulti,Human_voice,/,train\n"
    # Duplicate IDs are rejected rather than silently choosing one row.
    with pytest.raises(ValueError, match="duplicate"):
        build_split_rows(source, metadata, excluded_ids=set(), excluded_uploaders=set())

    one_row = 'fname,labels,mids,split\nmulti,"Animal,Human_voice",/,train\n'
    split = build_split_rows(one_row, metadata, excluded_ids=set(), excluded_uploaders=set())
    row = next(row for rows in split.values() for row in rows)
    assert row["targets"]["Animal"] is True
    assert row["targets"]["Human_voice"] is True
    assert tuple(row["targets"]) == TARGET_LABELS
    assert row["source_url"].endswith("/alice/sounds/multi/")


def test_validation_gate_requires_examples_and_independent_uploaders():
    splits = {
        "train": [{"sample_id": "a", "group_id": "train-user"}],
        "calibration": [{"sample_id": "b", "group_id": "cal-user"}],
        "validation": [{"sample_id": "c", "group_id": "val-user"}],
    }
    with pytest.raises(ValueError, match="1000-example/200-uploader"):
        validate_splits(splits)


def test_summary_counts_uploaders_and_hashes_stable_order():
    rows = [
        {
            "sample_id": "1",
            "source_id": "freesound:1",
            "group_id": "alice",
            "targets": {"Animal": True, "Human_voice": False, "Vehicle": False, "Music": False},
        },
        {
            "sample_id": "2",
            "source_id": "freesound:2",
            "group_id": "bob",
            "targets": {"Animal": False, "Human_voice": False, "Vehicle": True, "Music": False},
        },
    ]
    result = summarize_split(rows)
    assert result["examples"] == 2
    assert result["unique_uploaders"] == 2
    assert result["positives_by_label"]["Animal"] == {"examples": 1, "positive_uploaders": 1}
    assert result["ordered_sample_ids_sha256"] == summarize_split(rows)["ordered_sample_ids_sha256"]
