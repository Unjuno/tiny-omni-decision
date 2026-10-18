from scripts.build_greatest_hits_av_match import _group_id, build_probe


def _fixture():
    train = [f"2020-01-01-12-00-{index:02d}" for index in range(10)]
    annotations = {parent: [{"fields": ["metal", "hit", "static"]}] for parent in train}
    archive = []
    for parent in train:
        archive.extend(
            [
                {"name": f"{parent}_denoised.wav", "bytes": 100, "compressed": 80},
                {"name": f"{parent}_denoised.mp4", "bytes": 100, "compressed": 80},
            ]
        )
    partition = {"train": train, "validation": [], "quarantine": []}
    return partition, annotations, [], archive


def test_group_identity_keeps_timestamp_seconds_and_strips_only_seventh_component():
    assert _group_id("2015-02-21-17-57-20") == "2015-02-21-17-57-20"
    assert _group_id("2015-10-03-14-50-14-87") == "2015-10-03-14-50-14"


def test_pair_matching_is_deterministic_and_has_balanced_unimodal_marginals():
    data = _fixture()
    rows, summary = build_probe(*data)
    rows_again, summary_again = build_probe(*data)

    assert rows == rows_again
    assert summary["sample_id_order_sha256"] == summary_again["sample_id_order_sha256"]
    assert summary["pair_marginals_balanced"] is True
    assert summary["group_overlap"] == 0
    assert summary["rows"] == {"train": 12, "calibration": 4, "validation": 4}
    assert summary["selected_media_files"] == 20


def test_reviewed_source_prefixes_are_excluded_from_every_split():
    partition, annotations, _, archive = _fixture()
    reviewed = [{"parent": "2020-01-01-12-00-00"}]

    rows, summary = build_probe(partition, annotations, reviewed, archive)

    assert summary["reviewed_source_group_exclusions"] == ["2020-01-01-12-00-00"]
    assert all(
        "2020-01-01-12-00-00" not in row["audio_source_group"]
        and "2020-01-01-12-00-00" not in row["video_source_group"]
        for split_rows in rows.values()
        for row in split_rows
    )


def test_missing_media_fails_closed():
    partition, annotations, reviewed, archive = _fixture()
    archive = [member for member in archive if not member["name"].endswith("_denoised.mp4")]

    try:
        build_probe(partition, annotations, reviewed, archive)
    except ValueError as error:
        assert "missing media" in str(error)
    else:
        raise AssertionError("missing video assets must fail closed")


def test_media_resource_caps_fail_closed():
    partition, annotations, reviewed, archive = _fixture()
    archive[0]["bytes"] = 41 * 1024 * 1024

    try:
        build_probe(partition, annotations, reviewed, archive)
    except ValueError as error:
        assert "per-member" in str(error)
    else:
        raise AssertionError("oversized media members must fail closed")
