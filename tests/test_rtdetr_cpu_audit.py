import hashlib
import json
import struct

import pytest

from scripts.audit_rtdetr_cpu_candidate import inspect_checkpoint


def test_model_card_license_and_utf8_are_explicit(tmp_path):
    from scripts.audit_rtdetr_cpu_candidate import read_model_card

    card = tmp_path / "README.md"
    card.write_bytes("---\nlicense: apache-2.0\n---\nModel’s details".encode())
    assert read_model_card(card)["license"] == "apache-2.0"
    card.write_bytes(b"---\nlicense: unknown\n---\n")
    with pytest.raises(ValueError):
        read_model_card(card)


def test_config_bytes_must_match_immutable_git_blob_pin(tmp_path):
    from scripts.audit_rtdetr_cpu_candidate import verify_git_blob

    p = tmp_path / "config.json"
    original = b'{"size":640}'
    p.write_bytes(original)
    expected = hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest()
    assert verify_git_blob(p, expected) == hashlib.sha256(original).hexdigest()
    p.write_bytes(b'{"size":320}')
    with pytest.raises(ValueError):
        verify_git_blob(p, expected)


def write_weight(tmp_path, header, payload):
    p = tmp_path / "model.safetensors"
    encoded = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_checkpoint_hash_header_shapes_dtypes_and_exact_payload_accounting(tmp_path):
    p, pin = write_weight(
        tmp_path, dict(weight=dict(dtype="F32", shape=[2, 3], data_offsets=[0, 24])), b"\0" * 24
    )
    r = inspect_checkpoint(p, pin)
    assert r["tensor_elements"] == 6
    assert r["tensor_data_bytes"] == 24
    assert r["dtype_elements"] == {"F32": 6}
    with pytest.raises(ValueError):
        inspect_checkpoint(p, "a" * 64)


def test_bad_overlapping_tensor_spans_or_missing_payload_are_rejected(tmp_path):
    header = dict(
        a=dict(dtype="F32", shape=[1], data_offsets=[0, 4]),
        b=dict(dtype="F32", shape=[1], data_offsets=[0, 4]),
    )
    p, pin = write_weight(tmp_path, header, b"\0" * 8)
    with pytest.raises(ValueError):
        inspect_checkpoint(p, pin)
    p, pin = write_weight(
        tmp_path, dict(a=dict(dtype="F32", shape=[2], data_offsets=[0, 8])), b"\0" * 4
    )
    with pytest.raises(ValueError):
        inspect_checkpoint(p, pin)
