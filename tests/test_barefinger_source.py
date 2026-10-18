from itertools import product

import pytest

TEXTURES = ["7t", "10t", "18t", "39t", "44t", "54t", "59t", "83t", "108t", "120t"]
TRIALS = ["std_0", "std_1", "std_2", "cte_force_3", "cte_speed_4"]


def row(name):
    return dict(
        name=name,
        directory=False,
        size=8,
        compressed_size=6,
        compression=8,
        crc32="1234abcd",
        offset=4,
    )


def fixture():
    trials = [
        row(f"datah5/subject_{s}/{t}/subject_{s}_{t}_{r}.h5")
        for s, t, r in product(range(10), TEXTURES, TRIALS)
    ]
    trials.append(row("datah5/texture_metadata.json"))
    images = [
        row(f"img/{t}/{t}_{c}_{n}.jpg") for t, c, n in product(TEXTURES, ["l", "r"], range(10))
    ]
    return trials, images


def test_complete_cohort_distinguishes_participant_and_shared_photo_identity():
    from scripts.audit_barefinger_source import account_directories

    data, images = fixture()
    report = account_directories(data, images)
    assert report["physical_trials"] == 500
    assert report["participant_texture_groups"] == 100
    assert report["texture_photo_groups"] == 10
    assert report["stereo_photos"] == 200
    assert report["images_independent_by_participant"] is False
    assert report["native_rgb_video_members"] == 0
    assert report["training_admission"] == "NOT_ADMITTED"


@pytest.mark.parametrize("change", ["missing", "duplicate", "unsafe", "mismatch"])
def test_trial_integrity_fails_closed(change):
    from scripts.audit_barefinger_source import account_directories

    data, images = fixture()
    if change == "missing":
        data.pop(0)
    elif change == "duplicate":
        data.append(data[0])
    elif change == "unsafe":
        data[0] = row("datah5/../subject_0.h5")
    else:
        data[0] = row("datah5/subject_0/7t/subject_1_7t_std_0.h5")
    with pytest.raises(ValueError):
        account_directories(data, images)


def test_photo_alias_missing_and_unknown_members_fail_closed():
    from scripts.audit_barefinger_source import account_directories

    data, images = fixture()
    for altered in [images[:-1], images + [images[0]], images + [row("img/7t/7t_l_10.jpg")]]:
        with pytest.raises(ValueError):
            account_directories(data, altered)
    with pytest.raises(ValueError):
        account_directories(data + [row("datah5/video.mp4")], images)


def test_unbound_version_must_not_be_parsed(tmp_path, monkeypatch):
    from scripts import audit_barefinger_source as module

    monkeypatch.setattr(
        module,
        "read_bound_json",
        lambda *args: dict(
            status="PASS_IMMUTABLE_LICENSE_README_AND_DIRECTORY_ONLY",
            media_members_opened=False,
            training=False,
            full_archive_md5_verified=False,
            requests=[],
        ),
    )
    monkeypatch.setattr(module, "recheck", lambda *args: None)
    (tmp_path / "version-1.json").write_text("invalid-json")
    with pytest.raises(ValueError, match="required pinned source"):
        module.build(tmp_path, "0" * 64)


def test_failed_terminal_recheck_does_not_publish_pass(tmp_path, monkeypatch):
    from scripts import audit_barefinger_source as module

    output = tmp_path / "result.json"
    monkeypatch.setattr(module, "build", lambda *args: dict(validated_hashes={}))
    monkeypatch.setattr(module, "peak_bytes", lambda: 1024)
    monkeypatch.setattr(
        module, "recheck", lambda *args: (_ for _ in ()).throw(ValueError("source changed"))
    )
    with pytest.raises(ValueError, match="source changed"):
        module.execute(tmp_path, "0" * 64, output)
    assert not output.exists()
    prepared = output.with_suffix(".accounting.json")
    assert prepared.exists()
    assert "PASS_" not in prepared.read_text()


def test_existing_report_is_preserved(tmp_path):
    from scripts.audit_barefinger_source import execute

    output = tmp_path / "result.json"
    output.write_text("protected")
    with pytest.raises(ValueError, match="new source report"):
        execute(tmp_path, "0" * 64, output)
    assert output.read_text() == "protected"
