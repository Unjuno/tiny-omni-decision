from scripts.freeze_inaturalist_cc0_bird_metadata import build_manifests, observer_group_hash


def _obs(observation_id, observer_id, taxon_id, label, photos, observation_license="cc0"):
    return (
        label,
        taxon_id,
        {
            "id": observation_id,
            "license_code": observation_license,
            "user": {"id": observer_id, "login": "must-not-leak"},
            "location": "1,2",
            "description": "must-not-leak",
            "photos": photos,
        },
    )


def _photo(photo_id, license_code="cc0"):
    return {"id": photo_id, "license_code": license_code, "url": f"https://example.test/{photo_id}.jpg"}


def test_strict_cc0_filter_and_privacy_minimized_rows():
    rows = [
        _obs("obs-a", "user-a", 1, "A", [_photo("photo-a"), _photo("photo-denied", "cc-by-nc")]),
        _obs("obs-b", "user-b", 1, "A", [_photo("photo-b")], observation_license="cc-by"),
        _obs("obs-c", "user-c", 1, "A", [_photo("photo-missing", None)]),
    ]

    manifests, counts = build_manifests(rows, seed=17)

    emitted = manifests["train"] + manifests["validation"]
    assert len(emitted) == 1
    assert emitted[0]["photo_id"] == "photo-a"
    assert emitted[0]["photo_license"] == "cc0"
    assert emitted[0]["observation_license"] == "cc0"
    assert "observer_group_sha256" in emitted[0]
    assert "observer_id" not in emitted[0]
    assert "login" not in str(emitted[0])
    assert "location" not in str(emitted[0])
    assert counts["observation_license_rejected"] == 1
    assert counts["photo_objects_non_cc0_or_missing"] == 2


def test_observer_group_is_global_and_only_one_row_per_class_group():
    rows = [
        _obs("obs-a", "shared-user", 1, "A", [_photo("photo-a")]),
        _obs("obs-b", "shared-user", 1, "A", [_photo("photo-b")]),
        _obs("obs-c", "shared-user", 2, "B", [_photo("photo-c")]),
    ]

    manifests, _ = build_manifests(rows, seed=17, train_percent=70)
    emitted = manifests["train"] + manifests["validation"]

    assert len(emitted) == 2
    assert {row["taxon_id"] for row in emitted} == {1, 2}
    assert len({row["observer_group_sha256"] for row in emitted}) == 1
    assert len({row["split"] for row in emitted}) == 1
    assert len({row["observation_id"] for row in emitted}) == 2


def test_photo_id_reused_across_taxa_is_excluded():
    rows = [
        _obs("obs-a", "user-a", 1, "A", [_photo("shared"), _photo("alternate-a")]),
        _obs("obs-b", "user-b", 2, "B", [_photo("shared"), _photo("alternate-b")]),
    ]

    manifests, counts = build_manifests(rows, seed=17)
    emitted = manifests["train"] + manifests["validation"]

    assert counts["cross_taxon_photo_ids"] == 1
    assert "shared" not in {row["photo_id"] for row in emitted}
    assert len(emitted) == 2


def test_cross_taxon_observation_id_is_excluded():
    rows = [
        _obs("same-observation", "user-a", 1, "A", [_photo("photo-a")]),
        _obs("same-observation", "user-a", 2, "B", [_photo("photo-b")]),
    ]

    manifests, counts = build_manifests(rows, seed=17)

    assert counts["cross_taxon_observation_ids"] == 1
    assert not manifests["train"]
    assert not manifests["validation"]


def test_group_split_is_deterministic_and_distinct_between_observers():
    rows = [
        _obs("obs-a", "user-a", 1, "A", [_photo("photo-a")]),
        _obs("obs-b", "user-b", 1, "A", [_photo("photo-b")]),
    ]

    first, _ = build_manifests(rows, seed=17)
    second, _ = build_manifests(rows, seed=17)
    assert first == second
    assert observer_group_hash("user-a", 17) != observer_group_hash("user-b", 17)