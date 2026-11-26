import hashlib
import json

import numpy as np
import pytest


def test_native_resample_preserves_duration_and_rejects_invalid_pcm():
    pytest.importorskip("av")
    from scripts.robomnist_audio_cue import resample_native

    t = np.arange(44100, dtype=np.float32) / 44100
    pcm = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[:, None]
    actual = resample_native(pcm, 44100)
    assert actual.shape == (16000,)
    assert actual.dtype == np.float32
    assert abs(float(np.sqrt(np.mean(actual[100:-100] ** 2))) - 0.2 / np.sqrt(2)) < 1e-3
    assert np.isfinite(actual).all()
    assert np.all(resample_native(np.zeros_like(pcm), 44100) == 0)
    for bad in [pcm.astype(np.float64), pcm[:, 0], pcm[:0], np.full_like(pcm, np.nan)]:
        with pytest.raises(ValueError):
            resample_native(bad, 44100)
    with pytest.raises(ValueError):
        resample_native(pcm, 16000)


def test_fixed_reference_hash_is_required_before_parsing(tmp_path):
    from scripts.robomnist_audio_cue import read_bound_json, recheck

    p = tmp_path / "reference.json"
    body = json.dumps({"key": "value"}).encode()
    p.write_bytes(body)
    h = hashlib.sha256(body).hexdigest()
    assert read_bound_json(p, h) == {"key": "value"}
    recheck({str(p): h})
    p.write_bytes(body + b" ")
    with pytest.raises(ValueError):
        read_bound_json(p, h)
    with pytest.raises(ValueError):
        recheck({str(p): h})


def test_cue_description_never_infers_truth_or_clock_from_asr():
    from scripts.robomnist_audio_cue import describe_numbers

    r = describe_numbers("Zero, one two 3; fourteen fifteen, 24 robots.")
    assert r["observations_0_to_20"] == [0, 1, 2, 3, 14, 15]
    assert r["speech_ground_truth"] == "UNVERIFIED"
    assert r["collection_counter_active"] == "UNKNOWN"
    assert r["common_clock"] == "UNVERIFIED"


def test_generation_must_have_exact_prefix_and_terminal_eos_without_cap_rescue():
    from scripts.robomnist_audio_cue import require_sequence

    ids = np.array([50258, 50259, 50359, 50363, 10, 50257], np.int64)
    require_sequence(ids)
    for bad in [
        ids[:-1],
        np.concatenate([ids[:4], np.ones(129, np.int64), [50257]]),
        ids.astype(np.int32),
        np.array([1, 50257], np.int64),
    ]:
        with pytest.raises(ValueError):
            require_sequence(bad)


def test_frozen_audio_scope_rejects_model_updates_or_gpu():
    from scripts.probe_robomnist_audio_cue import require_audio_scope

    r = dict(
        parameters=37760640,
        trainable_parameters=0,
        model_training=False,
        gradients_present=False,
        cuda_initialized=False,
        state_before="a" * 64,
        state_after="a" * 64,
        forward_cpu_rng_unchanged=True,
        generated_conditions=6,
    )
    require_audio_scope(r)
    for change in [
        dict(trainable_parameters=1),
        dict(state_after="b" * 64),
        dict(cuda_initialized=True),
        dict(generated_conditions=5),
    ]:
        with pytest.raises(ValueError):
            require_audio_scope({**r, **change})


def test_failure_records_unknown_monitor_and_does_not_overwrite(tmp_path, monkeypatch):
    from scripts import probe_robomnist_audio_cue as module

    def fail(*args):
        raise RuntimeError("injected source fault before model")

    monkeypatch.setattr(module, "run_body", fail)
    monkeypatch.setattr(module, "failure_peak", lambda: None)
    out = tmp_path / "new"
    assert module.execute(out) is False
    r = json.loads((out / "result.json").read_bytes())
    assert r["status"] == "FAIL_INCOMPLETE_ACOUSTIC_CUE_DIAGNOSTIC"
    assert r["peak_cpu_bytes"] is None
    assert r["decision_quality_measured"] is False
    with pytest.raises(ValueError):
        module.execute(out)


def test_all_prepared_arrays_and_records_are_mandatory(tmp_path):
    from scripts.diagnose_robomnist_lengths import write, write_arrays
    from scripts.probe_robomnist_audio_cue import require_products

    c, r, a = (
        {"scope": "fixture"},
        [{"condition": "fixture"}],
        {"wave": np.arange(4, dtype=np.float32)},
    )
    pins = {
        "config.json": write(tmp_path / "config.json", c),
        "transcripts.json": write(tmp_path / "transcripts.json", r),
        "arrays.npz": write_arrays(tmp_path / "arrays.npz", a),
    }
    require_products(tmp_path, pins, c, r, a)
    with pytest.raises(ValueError):
        require_products(tmp_path, pins, c, r, {"wave": a["wave"] + 1})
    (tmp_path / "transcripts.json").write_bytes(b"[]")
    with pytest.raises(ValueError):
        require_products(tmp_path, pins, c, r, a)


def test_fixed_decoder_returns_full_unstripped_sequence_for_prefix_eos_checks():
    from scripts.probe_robomnist_audio_cue import GENERATION

    assert GENERATION["return_dict_in_generate"] is True


def test_actual_whisper_and_generic_generation_implementations_are_pinned():
    pytest.importorskip("torch")
    from scripts.probe_robomnist_audio_cue import generation_paths

    names = {p.name for p in generation_paths()}
    assert {
        "generation_whisper.py",
        "utils.py",
        "logits_process.py",
        "stopping_criteria.py",
        "configuration_utils.py",
    } <= names
