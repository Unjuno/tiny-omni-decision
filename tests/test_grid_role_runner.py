import hashlib
import json

import pytest

from scripts import prepare_grid_role_corpus as runner


def fixture_rows():
    rows = [
        dict(video_id="one", split="train", media_sha256="a"),
        dict(video_id="two", split="train", media_sha256="b"),
        dict(video_id="dev", split="valid", media_sha256="c"),
    ]
    clips = [dict(video_id="one", sha256="a"), dict(video_id="two", sha256="b")]
    return rows, clips


def test_only_train_rows_selected_without_loading_development_arrays():
    rows, clips = fixture_rows()
    selected, lookup = runner.admit_train_rows(rows, clips, expected=2)
    assert selected == rows[:2] and list(lookup) == ["one", "two"]


@pytest.mark.parametrize("kind", ["id", "hash", "missing", "manifest"])
def test_train_identity_content_or_inventory_mismatch_fails_closed(kind):
    rows, clips = fixture_rows()
    if kind == "id":
        rows[1]["video_id"] = "one"
    elif kind == "hash":
        rows[1]["media_sha256"] = "a"
    elif kind == "missing":
        rows.pop(0)
    else:
        clips[1]["sha256"] = "changed"
    with pytest.raises(ValueError):
        runner.admit_train_rows(rows, clips, expected=2)


@pytest.mark.parametrize("key,value", [("video_id", "one"), ("media_sha256", "a")])
def test_metadata_cross_split_identity_or_content_overlap_rejected(key, value):
    rows, clips = fixture_rows()
    rows[-1][key] = value
    with pytest.raises(ValueError):
        runner.admit_train_rows(rows, clips, expected=2)


def test_fresh_reference_requires_complete_result_and_actual_config_binding(tmp_path):
    cfg = tmp_path / "effective-config.json"
    cfg.write_bytes(b"fixed config")
    digest = hashlib.sha256(cfg.read_bytes()).hexdigest()
    result = dict(
        status="PASS_ALL_TRAIN_GRID_ROLE_CORPUS_NOT_QUALITY",
        config_sha256=digest,
        clips=99,
        rows=[dict(video_id=str(i)) for i in range(99)],
    )
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert runner.require_reference(tmp_path, digest)["config_sha256"] == digest
    cfg.write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.require_reference(tmp_path, digest)
    cfg.write_bytes(b"fixed config")
    result["status"] = "FAIL_INCOMPLETE"
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.require_reference(tmp_path, digest)


def test_effective_config_mutation_rejected_before_success(tmp_path):
    path = tmp_path / "effective-config.json"
    path.write_bytes(b"fixed")
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    runner.verify_effective_config(path, pin)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.verify_effective_config(path, pin)
