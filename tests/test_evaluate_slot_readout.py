import json

import pytest

pytest.importorskip("torch")
from scripts.evaluate_slot_readout import load_valid  # noqa: E402


def test_incomplete_validation_rejected(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"status": "RUNNING"}))
    with pytest.raises(ValueError, match="complete"):
        load_valid(tmp_path)


def test_other_protocol_rejected(tmp_path):
    from scripts.audit_correspondence_validation import sha

    path = tmp_path / "protocol.json"
    path.write_text("{}")
    (tmp_path / "result.json").write_text(json.dumps(dict(
        status="PASS_FIXED_VALID_SLOT_EXTRACTION", weights_unchanged=True,
        protocol_sha256=sha(path),
    )))
    with pytest.raises(ValueError, match="experiment protocol"):
        load_valid(tmp_path)
