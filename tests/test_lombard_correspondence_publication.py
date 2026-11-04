import hashlib

import pytest


def test_publication_hash_gate_rejects_empty_and_mutated_payloads(tmp_path):
    from scripts.report_lombard_correspondence import verify

    with pytest.raises(ValueError):
        verify({})
    path = tmp_path / "catalog.npy"
    path.write_bytes(b"fixed")
    pins = {str(path): hashlib.sha256(b"fixed").hexdigest()}
    verify(pins)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify(pins)


def test_publication_rejects_conflicting_source_authorities():
    from scripts.report_lombard_correspondence import merge_pins

    pins = {"same": "a"}
    merge_pins(pins, {"new": "b", "same": "a"})
    with pytest.raises(ValueError):
        merge_pins(pins, {"same": "c"})
