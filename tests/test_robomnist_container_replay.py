import hashlib
import json

import pytest


def test_report_is_hashed_before_parse_and_terminal_mutation_is_rejected(tmp_path):
    from scripts.replay_robomnist_container_features import read_report, recheck

    path = tmp_path / "result.json"
    body = json.dumps({"status": "fixture"}).encode()
    path.write_bytes(body)
    value, expected = read_report(path, hashlib.sha256(body).hexdigest())
    assert value == {"status": "fixture"}
    recheck({str(path): expected})
    path.write_bytes(body + b" ")
    with pytest.raises(ValueError):
        read_report(path, expected)
    with pytest.raises(ValueError):
        recheck({str(path): expected})


def test_required_source_cohort_cannot_be_dropped_or_repinned():
    from scripts.replay_robomnist_container_features import require_cohort

    require_cohort({"source": "a" * 64}, {"source": "a" * 64, "library": "b" * 64})
    with pytest.raises(ValueError):
        require_cohort({"source": "a" * 64}, {})
    with pytest.raises(ValueError):
        require_cohort({"source": "a" * 64}, {"source": "b" * 64})


def test_replay_rejects_training_gpu_or_quality_scope():
    from scripts.replay_robomnist_container_features import require_scope

    r = dict(
        status="PASS_NATIVE_CONTAINER_AND_FROZEN_IMAGE_FEATURE_MECHANICS_ONLY",
        group="Ar1_Vhigh_Ac0_Rp01_UNC062",
        unique_source_recordings=1,
        decoded_frames=1350,
        projected_image_observations=18,
        encoded_image_instances=54,
        encoder_forward_calls=18,
        model_training=False,
        teacher_training=False,
        new_source_payloads=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        quality_measured=False,
        native_temporal_model=False,
        gpu_used=False,
        training_corpus_admission="NOT_ADMITTED",
        common_clock="UNVERIFIED",
        elapsed_seconds=25,
        peak_cpu_bytes=1200000000,
    )
    require_scope(r)
    for k in ["quality_measured", "gpu_used", "teacher_training"]:
        with pytest.raises(ValueError):
            require_scope({**r, k: True})
    with pytest.raises(ValueError):
        require_scope({**r, "decoded_frames": 450})


def test_actual_derivative_prefix_and_fixed_atom_survive_rehash_attempt(tmp_path):
    from scripts.probe_robomnist_container_features import sha
    from scripts.replay_robomnist_container_features import verify_atom
    from scripts.robomnist_container_features import append_free_atom

    source, derived = tmp_path / "source.mp4", tmp_path / "derived.mp4"
    source.write_bytes(b"fixed-source-prefix")
    row = append_free_atom(source, derived)
    verify_atom(source, derived, row)
    data = bytearray(derived.read_bytes())
    data[0] ^= 1
    derived.write_bytes(data)
    # Even substituting the changed product hash cannot hide a modified prefix.
    with pytest.raises(ValueError, match="prefix"):
        verify_atom(source, derived, {**row, "derived_sha256": sha(derived)})
    data[0] ^= 1
    data[-1] = 1
    derived.write_bytes(data)
    with pytest.raises(ValueError, match="zero payload"):
        verify_atom(source, derived, {**row, "derived_sha256": sha(derived)})


def test_consumer_snapshot_preserves_prework_hashes(tmp_path):
    from scripts.replay_robomnist_container_features import consumer_snapshot, recheck

    p = tmp_path / "consumer.py"
    p.write_bytes(b"original code")
    pins = consumer_snapshot([p])
    assert pins == {str(p.resolve()): hashlib.sha256(b"original code").hexdigest()}
    p.write_bytes(b"edited during long work")
    with pytest.raises(ValueError):
        recheck(pins)
