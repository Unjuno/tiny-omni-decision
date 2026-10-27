from scripts.join_inat_photo_metadata import reconcile_candidate


def candidate():
    return {"observation_id": 1, "photo_id": 7, "photo_license": "cc-by",
            "observer_key": "observer"}


def test_exact_uuid_license_and_photographer_are_required():
    mapping = {"observation_id": 1, "photo_id": 7, "status": "MATCH",
               "observation_uuid": "uuid"}
    metadata = [{"observation_uuid": "uuid", "photographer_key": "a" * 64,
                 "license": "CC-BY", "photo_uuid": "photo-uuid"}]
    result = reconcile_candidate(candidate(), mapping, metadata)
    assert result["status"] == "MATCH" and result["photographer_key"] == "a" * 64
    metadata[0]["license"] = "CC-BY-NC"
    assert "license_mismatch" in reconcile_candidate(candidate(), mapping, metadata)["reasons"]


def test_multiple_or_missing_source_rows_are_quarantined():
    mapping = {"observation_id": 1, "photo_id": 7, "status": "MATCH",
               "observation_uuid": "uuid"}
    row = {"observation_uuid": "uuid", "photographer_key": "x",
           "license": "CC-BY", "photo_uuid": "p"}
    outcome = reconcile_candidate(candidate(), mapping, [row, row])
    assert "source_row_ambiguous" in outcome["reasons"]
    assert "source_row_missing" in reconcile_candidate(candidate(), mapping, [])["reasons"]
