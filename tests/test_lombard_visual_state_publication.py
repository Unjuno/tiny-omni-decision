import pytest


def test_publication_requires_inputs_and_rejects_mutated_payload(tmp_path):
    from scripts.report_lombard_visual_state import verify

    with pytest.raises(ValueError):
        verify({})
    payload = tmp_path / "changed.npz"
    payload.write_bytes(b"unverified")
    with pytest.raises(ValueError):
        verify({str(payload): "0" * 64})
