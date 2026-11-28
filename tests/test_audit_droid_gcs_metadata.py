from scripts.audit_droid_gcs_metadata import (
    _choose_sample,
    _classify_path,
    _group_fields,
)


def _item(lab, status, episode, *, generation=1):
    return {
        "name": (
            f"robotics/droid_raw/1.0.1/{lab}/{status}/2023-01-01/episode-{episode}/"
            f"metadata_{lab}-{episode}.json"
        ),
        "generation": str(generation),
        "size": "100",
        "md5Hash": "dGVzdA==",
    }


def test_classify_only_versioned_raw_success_and_failure_metadata_paths():
    assert _classify_path(_item("LabA", "success", 1)["name"]) == ("LabA", "success")
    assert _classify_path(_item("LabA", "failure", 2)["name"]) == ("LabA", "failure")
    assert _classify_path("robotics/droid_raw/1.0.1/LabA/success/episode/video.mp4") is None
    assert _classify_path("robotics/droid_raw/1.0.1/LabA/unknown/episode/metadata.json") is None


def test_hash_sampling_is_deterministic_and_stratified_by_lab_and_path_status():
    objects = [
        _item(lab, status, episode)
        for lab in ("LabA", "LabB")
        for status in ("success", "failure")
        for episode in range(5)
    ]
    first, totals = _choose_sample(objects, per_group=2)
    second, _ = _choose_sample(list(reversed(objects)), per_group=2)

    assert [item["name"] for item in first] == [item["name"] for item in second]
    assert len(first) == 8
    assert totals == {
        ("LabA", "success"): 5,
        ("LabA", "failure"): 5,
        ("LabB", "success"): 5,
        ("LabB", "failure"): 5,
    }


def test_group_fields_reject_schema_defaults_as_valid_split_keys():
    assert _group_fields({"building": "Building A", "scene_id": 4, "user_id": "user-1"}) == (
        True,
        True,
        True,
    )
    assert _group_fields({"building": "N/A", "scene_id": -1, "user_id": ""}) == (
        False,
        False,
        False,
    )
