import pytest


def fixture_rows():
    classes = [{"class_id": 0, "material_english": "brass"}]
    sessions = [
        dict(
            session_id=s,
            material_english="brass",
            force_n=f,
            raw_audio_filename=s + ".wav",
            raw_video_filename=s + ".avi",
            raw_tactile_filename=s + ".csv",
        )
        for s, f in [("a", "3"), ("b", "6")]
    ]
    samples = [
        dict(
            sample_id="parent" + s,
            session_id=s,
            material_english="brass",
            class_id="0",
            force_n=f,
            split=split,
            start_epoch="0",
            end_epoch="1",
        )
        for s, f, split in [("a", "3", "train"), ("b", "6", "test")]
    ]
    clips = [
        dict(
            clip_id=k,
            sample_id="parent" + s,
            session_id=s,
            material_english="brass",
            class_id="0",
            force_n=f,
            split=split,
            source_start_epoch=start,
            source_end_epoch=end,
            source_duration_s="0.5",
            stored_duration_s="2.0",
            stretch_factor="4.0",
            window_stride_s="0.15",
        )
        for k, s, f, split, start, end in [
            ("a1", "a", "3", "train", "0", "0.5"),
            ("a2", "a", "3", "train", "0.15", "0.65"),
            ("b1", "b", "6", "test", "0", "0.5"),
        ]
    ]
    return classes, sessions, samples, clips


def test_full_recording_grouping_and_overlap_not_inflated_independence():
    from scripts.audit_vistouch_metadata import audit_metadata

    r = audit_metadata(*fixture_rows())
    assert r["clips"] == 3
    assert r["source_sessions"] == 2
    assert r["cross_split_sessions"] == []
    assert r["split_counts"] == {"train": 2, "test": 1}
    assert r["independent_validation_present"] is False
    assert r["unique_source_seconds"] == pytest.approx(1.15)
    assert r["source_window_seconds_sum"] == 1.5
    assert r["stored_duration_seconds_sum"] == 6.0
    assert r["all_stretch_factors"] == [4.0]
    assert r["media_hash_overlap"] == "UNVERIFIED"
    assert r["specimen_identity"] == "UNKNOWN"
    assert r["training_corpus_admission"] == "NOT_ADMITTED"


def test_parent_label_or_interval_mismatch_fails_closed():
    from scripts.audit_vistouch_metadata import audit_metadata

    c, s, p, rows = fixture_rows()
    for change in [
        dict(class_id="1"),
        dict(source_end_epoch="2"),
        dict(source_start_epoch="0.8"),
        dict(sample_id="missing"),
        dict(force_n="9"),
    ]:
        with pytest.raises(ValueError):
            audit_metadata(c, s, p, [{**rows[0], **change}, *rows[1:]])


def test_alias_recording_and_mixed_split_are_reported_not_silently_admitted():
    from scripts.audit_vistouch_metadata import audit_metadata

    c, s, p, rows = fixture_rows()
    s[1]["raw_audio_filename"] = s[0]["raw_audio_filename"]
    r = audit_metadata(c, s, p, rows)
    assert len(r["cross_split_source_assets"]) == 1
    assert r["training_corpus_admission"] == "NOT_ADMITTED"
    # A parent and its child moved together still cannot split one source session.
    p.append({**p[0], "sample_id": "other", "split": "test"})
    rows.append({**rows[0], "clip_id": "a3", "sample_id": "other", "split": "test"})
    r = audit_metadata(c, s, p, rows)
    assert r["cross_split_sessions"] == ["a"]
    assert r["duplicate_source_windows"] == 1


def test_terminal_source_mutation_cannot_publish_pass(tmp_path, monkeypatch):
    import json

    from scripts import audit_vistouch_metadata as module

    output = tmp_path / "result.json"
    monkeypatch.setattr(
        module, "build", lambda *args: {"validated_hashes": {}, "quality_measured": False}
    )
    monkeypatch.setattr(
        module, "guard", lambda *args: dict(elapsed_seconds=0.1, peak_cpu_bytes=100)
    )

    def mutation(*args):
        raise ValueError("source changed after prepared readback")

    monkeypatch.setattr(module, "recheck", mutation)
    assert module.execute(output) is False
    r = json.loads(output.read_bytes())
    assert r["status"] == "FAIL_INCOMPLETE_SOURCE_METADATA_AUDIT"
    assert output.with_suffix(".accounting.json").exists()
    with pytest.raises(ValueError):
        module.execute(output)


def test_session_disjoint_validation_is_not_full_independence():
    from scripts.audit_vistouch_metadata import audit_metadata

    c, s, p, rows = fixture_rows()
    p[1]["split"] = "val"
    rows[2]["split"] = "val"
    s[1]["raw_audio_filename"] = s[0]["raw_audio_filename"]
    r = audit_metadata(c, s, p, rows)
    assert r["validation_sessions_disjoint"] is True
    assert r["independent_validation_present"] == "UNVERIFIED"
    assert r["cross_split_source_assets"]


def test_resource_cap_includes_prospective_terminal_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import audit_vistouch_metadata as module

    output = tmp_path / "result.json"
    output.with_suffix(".accounting.json").write_bytes(b"prepared")
    monkeypatch.setattr(module, "peak_bytes", lambda: 100)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=4 * 2**30))
    with pytest.raises(RuntimeError):
        module.guard(output, module.time.perf_counter(), 2**20)
