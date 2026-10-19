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


def test_holoassist_extraction_binds_the_selected_media_plan():
    _require_frozen_selection({"selection_sha256": HOLOASSIST_SELECTION_SHA256})
    with pytest.raises(ValueError, match="frozen HoloAssist selection"):
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
