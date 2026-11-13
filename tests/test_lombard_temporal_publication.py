import pytest


def test_publication_hash_gate_rejects_changed_control_payload(tmp_path):
    from scripts.publish_lombard_temporal_results_checked import verify

    path = tmp_path / "control.npz"
    path.write_bytes(b"not the authorized control")
    with pytest.raises(ValueError, match="publication input"):
        verify({str(path): "0" * 64})
