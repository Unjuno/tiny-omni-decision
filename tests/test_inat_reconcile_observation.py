import pytest

from scripts.reconcile_inat_observations import reconcile_row, validate_batch, validate_unique_uuids


def test_reconcile_pass_and_license_change():
    frozen = {"observation_id": 1, "observer_key": "key", "photo_id": 7,
              "photo_license": "cc-by", "attribution": "Photo by A",
              "observation_license": "cc-by", "observed_taxon_id": 10}
    valid_uuid = "12345678-1234-4234-8234-123456789abc"
    current = {"id": 1, "uuid": valid_uuid, "quality_grade": "research",
               "license_code": "cc-by", "user": {"id": 4}, "taxon": {"id": 10},
               "photos": [{"id": 7, "license_code": "cc-by", "hidden": False,
                           "attribution": "Photo by A"}]}
    result = reconcile_row(frozen, current, lambda _: "key")
    assert result["status"] == "MATCH" and result["observation_uuid"] == valid_uuid
    current["photos"][0]["license_code"] = "cc-by-nc"
    result = reconcile_row(frozen, current, lambda _: "key")
    assert result["status"] == "QUARANTINE" and "photo_license_changed" in result["reasons"]


def test_malformed_uuid_is_quarantined():
    frozen = {"observation_id": 1, "observer_key": "key", "photo_id": 7,
              "photo_license": "cc-by", "attribution": "A",
              "observation_license": "cc-by", "observed_taxon_id": 10}
    current = {"id": 1, "uuid": "not-uuid", "quality_grade": "research",
               "license_code": "cc-by", "user": {"id": 4}, "taxon": {"id": 10},
               "photos": [{"id": 7, "license_code": "cc-by", "hidden": False,
                           "attribution": "A"}]}
    result = reconcile_row(frozen, current, lambda _: "key")
    assert result["status"] == "QUARANTINE" and "invalid_uuid" in result["reasons"]


def test_missing_current_observation_is_retained():
    result = reconcile_row({"observation_id": 2, "photo_id": 5}, None, lambda _: "key")
    assert result["status"] == "QUARANTINE" and result["reasons"] == ["missing_observation"]


def test_batch_identity_checks_order_and_response_hash():
    rows = [{"observation_id": 1, "photo_id": 4}, {"observation_id": 2, "photo_id": 5}]
    batch = {"response_sha256": "a" * 64, "results": rows}
    assert validate_batch(rows, batch) == rows
    with pytest.raises(ValueError, match="identity"):
        validate_batch(list(reversed(rows)), batch)
    with pytest.raises(ValueError, match="digest"):
        validate_batch(rows, {"response_sha256": "no", "results": rows})


def test_duplicate_uuid_rejected():
    with pytest.raises(ValueError, match="duplicate UUID"):
        validate_unique_uuids([{"observation_uuid": "a"}, {"observation_uuid": "a"}])
