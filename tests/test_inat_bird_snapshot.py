import pytest

from scripts.snapshot_inat_birds import (
    clean_record,
    collision_summary,
    partition_photo_collisions,
    validate_snapshot,
)


def observation(observer=42, taxon=6930):
    return {
        "id": 10,
        "quality_grade": "research",
        "license_code": "cc-by",
        "user": {"id": observer},
        "taxon": {"id": taxon, "ancestry": "1/6930"},
        "photos": [
            {"id": 12, "license_code": "cc-by", "attribution": "A",
             "hidden": False, "url": "https://example.org/a.jpg"}
        ],
    }


def test_descendant_and_observer_split_are_stable():
    a = clean_record(observation(), 6930)
    b = clean_record(observation(taxon=999), 6930)
    assert a["split"] == b["split"]
    assert b["class_taxon_id"] == 6930
    assert b["observed_taxon_id"] == 999
    assert "user" not in a


def test_photo_and_observation_licenses_fail_closed():
    row = observation()
    row["photos"][0]["license_code"] = "cc-by-nc"
    assert clean_record(row, 6930) is None
    row = observation()
    row["license_code"] = "cc-by-nc"
    assert clean_record(row, 6930) is None


def test_duplicate_observation_rejected():
    row = clean_record(observation(), 6930)
    with pytest.raises(ValueError, match="duplicate observation"):
        validate_snapshot([row, row])


def test_duplicate_photo_rejected_separately():
    first = clean_record(observation(), 6930)
    other = observation(observer=43)
    other["id"] = 11
    second = clean_record(other, 6930)
    with pytest.raises(ValueError, match="duplicate photo"):
        validate_snapshot([first, second])


def test_collision_summary_detects_cross_split_and_conflicting_labels():
    first = clean_record(observation(), 6930)
    other = observation(observer=43)
    other["id"] = 11
    second = clean_record(other, 6930)
    second["split"] = "validation" if first["split"] == "train" else "train"
    second["class_taxon_id"] = 999
    result = collision_summary([first, second])
    assert result["duplicate_photo_ids"] == 1
    assert result["photo_groups_cross_split"] == 1
    assert result["photo_groups_conflicting_class"] == 1
    assert result["photo_groups_multiple_observers"] == 1


def test_all_rows_of_duplicate_photo_are_quarantined():
    first = clean_record(observation(), 6930)
    other = observation(observer=43)
    other["id"] = 11
    second = clean_record(other, 6930)
    third = clean_record(observation(observer=44), 6930)
    third["observation_id"] = 13
    third["photo_id"] = 14
    admitted, quarantine = partition_photo_collisions([first, second, third])
    assert admitted == [third]
    assert quarantine == [first, second]
    validate_snapshot(admitted)


def test_duplicate_observations_are_not_silently_quarantined():
    first = clean_record(observation(), 6930)
    with pytest.raises(ValueError, match="duplicate observation"):
        partition_photo_collisions([first, first])
