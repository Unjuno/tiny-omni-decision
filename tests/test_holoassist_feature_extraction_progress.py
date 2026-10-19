from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import scripts.extract_holoassist_frozen_features as extraction
from scripts.extract_holoassist_frozen_features import (
    HOLOASSIST_SELECTION_SHA256,
    WHISPER_ENCODER_SHA256,
    WHISPER_PROCESSOR_SHA256,
    WHISPER_SOURCE_WEIGHTS_SHA256,
    _load_progress,
    _require_frozen_selection,
    _save_progress,
    _validate_whisper_runtime,
)
from scripts.migrate_holoassist_feature_progress import migrate_progress_file


def test_holoassist_extraction_binds_the_selected_media_plan():
    _require_frozen_selection({"selection_sha256": HOLOASSIST_SELECTION_SHA256})
    _require_frozen_selection({"selection_sha256": "a" * 64}, "a" * 64)
    with pytest.raises(ValueError, match="explicitly pinned selection"):
        _require_frozen_selection({"selection_sha256": "b" * 64}, "a" * 64)
    with pytest.raises(ValueError, match="explicitly pinned selection"):
        _require_frozen_selection({"selection_sha256": "0" * 64})


def test_video_runtime_records_the_tensorflow_cpu_distribution(monkeypatch):
    calls = []

    def version(name):
        calls.append(name)
        if name == "tensorflow-cpu":
            return "2.18.1"
        raise extraction.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(extraction.importlib.metadata, "version", version)

    assert extraction.tensorflow_distribution_version() == "2.18.1"
    assert calls == ["tensorflow-cpu"]


def test_whisper_runtime_hashes_are_checked_before_extraction(tmp_path, monkeypatch):
    model_dir = tmp_path / "whisper-runtime"
    model_dir.mkdir()
    receipt = {
        "status": "PASS",
        "reference_revision": "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        "reference_files": {"model.safetensors": WHISPER_SOURCE_WEIGHTS_SHA256},
        "encoder_sha256": WHISPER_ENCODER_SHA256,
        "processor_sha256": WHISPER_PROCESSOR_SHA256,
    }
    (model_dir / "result.json").write_text(json.dumps(receipt), encoding="utf-8")
    (model_dir / "encoder.safetensors").touch()
    (model_dir / "preprocessor_config.json").touch()
    hashes = {
        "encoder.safetensors": WHISPER_ENCODER_SHA256,
        "preprocessor_config.json": WHISPER_PROCESSOR_SHA256,
    }
    monkeypatch.setattr(extraction, "sha256_file", lambda path: hashes[Path(path).name])

    assert _validate_whisper_runtime(model_dir) == model_dir.resolve()

    receipt["encoder_sha256"] = "0" * 64
    (model_dir / "result.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind to pinned source weights"):
        _validate_whisper_runtime(model_dir)


def test_cuda_feature_extraction_uses_shared_hotgpu_preflight(monkeypatch):
    calls = iter(("1234, 2, 16384\n", "123\n"))
    monkeypatch.setattr(extraction.subprocess, "check_output", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(extraction.os, "getpid", lambda: 456)
    observed = {}

    def preflight(**kwargs):
        observed.update(kwargs)
        return {"mode": "hotgpu_reserved", "reserved_mib": 3200}

    monkeypatch.setattr(extraction, "validate_gpu_preflight", preflight)
    slot = extraction.require_gpu_slot()

    assert slot == {"mode": "hotgpu_reserved", "reserved_mib": 3200}
    assert observed["memory_used_mib"] == 1234
    assert observed["utilization_pct"] == 2
    assert observed["total_mib"] == 16384
    assert observed["active_pids"] == {123}
    assert observed["own_pid"] == 456
    assert observed["environment"] is os.environ


def test_feature_progress_round_trips_and_binds_ids_and_fingerprint(tmp_path):
    path = tmp_path / "features.progress.npz"
    ids = ["train:session:0", "validation:session:0"]
    fingerprint = {"modality": "audio", "config_sha256": "a" * 64}
    features = np.arange(8, dtype=np.float32).reshape(2, 4)
    completed = np.asarray([True, False], dtype=np.bool_)

    _save_progress(
        path,
        ids=ids,
        features=features,
        completed=completed,
        fingerprint=fingerprint,
    )

    loaded, loaded_completed, resumed = _load_progress(
        path,
        ids=ids,
        width=4,
        fingerprint=fingerprint,
        resume=True,
    )

    assert resumed is True
    np.testing.assert_array_equal(loaded, features)
    np.testing.assert_array_equal(loaded_completed, completed)

    with pytest.raises(ValueError, match="different IDs, models, or provenance"):
        _load_progress(
            path,
            ids=list(reversed(ids)),
            width=4,
            fingerprint=fingerprint,
            resume=True,
        )

    with pytest.raises(ValueError, match="different IDs, models, or provenance"):
        _load_progress(
            path,
            ids=ids,
            width=4,
            fingerprint={**fingerprint, "config_sha256": "b" * 64},
            resume=True,
        )


def test_feature_progress_requires_explicit_resume_and_rejects_stale_temp(tmp_path):
    path = tmp_path / "features.progress.npz"
    ids = ["train:session:0"]
    fingerprint = {"modality": "video"}
    features = np.zeros((1, 3), dtype=np.float32)
    completed = np.zeros(1, dtype=np.bool_)

    _save_progress(
        path,
        ids=ids,
        features=features,
        completed=completed,
        fingerprint=fingerprint,
    )
    with pytest.raises(FileExistsError, match="pass --resume"):
        _load_progress(
            path,
            ids=ids,
            width=3,
            fingerprint=fingerprint,
            resume=False,
        )

    stale_temp = tmp_path / "stale.progress.npz.tmp"
    stale_temp.write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="temporary file needs inspection"):
        _save_progress(
            stale_temp.with_suffix(""),
            ids=ids,
            features=features,
            completed=completed,
            fingerprint=fingerprint,
        )


class _FakeVideoFrame:
    def __init__(self, pts, time_base, value, counter):
        self.pts = pts
        self.time_base = time_base
        self.value = value
        self.counter = counter

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        self.counter["conversions"] += 1
        return np.full((2, 3, 3), self.value, dtype=np.uint8)


class _FakeVideoContainer:
    def __init__(self, frames, stream):
        self.streams = [stream]
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def seek(self, *_args, **_kwargs):
        return None

    def decode(self, _stream):
        return iter(self.frames)


def _legacy_video_window_reference(frames, start, end):
    targets = np.linspace(start, end, num=8, endpoint=True)
    selected = []
    previous = None
    for frame in frames:
        if frame.pts is None:
            continue
        stamp = float(frame.pts * frame.time_base)
        if stamp < start:
            continue
        if stamp > end:
            if previous is not None:
                while len(selected) < 8:
                    selected.append(previous)
            break
        current = (stamp, frame.to_ndarray(format="rgb24"))
        while len(selected) < 8 and stamp >= targets[len(selected)]:
            selected.append(current)
        previous = current
        if len(selected) == 8:
            break
    if len(selected) < 8:
        if previous is None:
            raise ValueError("no frame in window")
        selected.extend([previous] * (8 - len(selected)))
    return np.stack([item[1] for item in selected])


def test_video_window_decoder_only_converts_selected_frames_and_preserves_pixels(monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    counter = {"conversions": 0}
    stream = SimpleNamespace(type="video", time_base=Fraction(1, 30))
    frames = [
        _FakeVideoFrame(index, stream.time_base, index % 255, counter) for index in range(901)
    ]
    container = _FakeVideoContainer(frames, stream)
    monkeypatch.setitem(
        __import__("sys").modules,
        "av",
        SimpleNamespace(open=lambda _path: container),
    )

    optimized = extraction.decode_video_window(Path("fake.mp4"), 0.0, 30.0)
    optimized_conversions = counter["conversions"]
    reference = _legacy_video_window_reference(frames, 0.0, 30.0)

    np.testing.assert_array_equal(optimized, reference)
    assert optimized_conversions == 8
    assert counter["conversions"] > 8


def test_video_window_decoder_keeps_end_padding_semantics(monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    counter = {"conversions": 0}
    stream = SimpleNamespace(type="video", time_base=Fraction(1, 1))
    frames = [_FakeVideoFrame(index, stream.time_base, index, counter) for index in range(9)]
    container = _FakeVideoContainer(frames, stream)
    monkeypatch.setitem(
        __import__("sys").modules,
        "av",
        SimpleNamespace(open=lambda _path: container),
    )

    optimized = extraction.decode_video_window(Path("fake.mp4"), 0.0, 10.0)
    reference = _legacy_video_window_reference(frames, 0.0, 10.0)

    np.testing.assert_array_equal(optimized, reference)


def test_video_window_decoder_repeats_last_frame_for_empty_tail_window(monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    counter = {"conversions": 0}
    stream = SimpleNamespace(type="video", time_base=Fraction(1, 1000))
    frames = [
        _FakeVideoFrame(1000, stream.time_base, 10, counter),
        _FakeVideoFrame(2000, stream.time_base, 20, counter),
        _FakeVideoFrame(2998, stream.time_base, 30, counter),
    ]
    container = _FakeVideoContainer(frames, stream)
    monkeypatch.setitem(
        __import__("sys").modules,
        "av",
        SimpleNamespace(open=lambda _path: container),
    )

    decoded = extraction.decode_video_window(Path("fake.mp4"), 3.0, 3.016)

    assert decoded.shape == (8, 2, 3, 3)
    np.testing.assert_array_equal(decoded, np.full((8, 2, 3, 3), 30, dtype=np.uint8))


def test_migrate_holoassist_progress_rebinds_only_approved_code_identity(tmp_path):
    from scripts.extract_holoassist_frozen_features import _save_progress

    ids = [f"window-{index}" for index in range(5)]
    features = np.zeros((5, 768), dtype=np.float32)
    features[:3] = 0.25
    completed = np.array([True, True, True, False, False])
    new_fingerprint = {
        "modality": "video",
        "device": "cpu",
        "config_sha256": "c" * 64,
        "extractor_source_commit": "new-commit",
        "extractor_script_sha256": "e" * 64,
    }
    old_fingerprint = {
        **new_fingerprint,
        "extractor_source_commit": "old-commit",
        "extractor_script_sha256": "d" * 64,
    }
    source = tmp_path / "attempt-03.progress.npz"
    destination = tmp_path / "attempt-04" / "video-features.progress.npz"
    _save_progress(
        source,
        ids=ids,
        features=features,
        completed=completed,
        fingerprint=old_fingerprint,
    )

    receipt = migrate_progress_file(
        source,
        destination,
        expected_ids=ids,
        expected_fingerprint=new_fingerprint,
        legacy_source_commit="old-commit",
        legacy_script_sha256="d" * 64,
    )

    assert receipt["status"] == "PASS"
    assert receipt["reused_completed_rows"] == 3
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sample_id"].tolist(), ids)
        np.testing.assert_array_equal(archive["features"], features)
        np.testing.assert_array_equal(archive["completed"], completed)
        assert json.loads(str(archive["fingerprint_json"].item())) == new_fingerprint


def test_migrate_holoassist_progress_rejects_unapproved_fingerprint_change(tmp_path):
    from scripts.extract_holoassist_frozen_features import _save_progress

    ids = ["window-0", "window-1"]
    features = np.zeros((2, 768), dtype=np.float32)
    features[0] = 0.5
    completed = np.array([True, False])
    expected = {
        "modality": "video",
        "device": "cpu",
        "config_sha256": "c" * 64,
        "extractor_source_commit": "new-commit",
        "extractor_script_sha256": "e" * 64,
    }
    legacy = {
        **expected,
        "config_sha256": "f" * 64,
        "extractor_source_commit": "old-commit",
        "extractor_script_sha256": "d" * 64,
    }
    source = tmp_path / "attempt-03.progress.npz"
    destination = tmp_path / "attempt-04" / "video-features.progress.npz"
    _save_progress(
        source,
        ids=ids,
        features=features,
        completed=completed,
        fingerprint=legacy,
    )

    with pytest.raises(ValueError, match="fingerprint differs beyond"):
        migrate_progress_file(
            source,
            destination,
            expected_ids=ids,
            expected_fingerprint=expected,
            legacy_source_commit="old-commit",
            legacy_script_sha256="d" * 64,
        )
    assert not destination.exists()
