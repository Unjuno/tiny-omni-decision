import hashlib
import json
import subprocess
import sys

import pytest


def test_inference_rejects_another_executing_checkout_before_input_access(tmp_path, monkeypatch):
    from scripts import evaluate_condition_relevance_transfer as target

    receipt = dict(
        status="PASS_FROZEN_CONDITION_RELEVANCE_COMPARISON",
        transfer_loaded=False, calibration_loaded=False,
        sealed_audit_loaded=False, model_loaded=False,
        source_root=str(tmp_path / "another-repo"), code_pins={},
    )
    monkeypatch.setattr(target, "_json", lambda *args: receipt)

    def unexpected(*args):
        pytest.fail("inputs or weights accessed before execution source binding")

    monkeypatch.setattr(target, "verify_pair_files", unexpected)
    with pytest.raises(ValueError, match="executing entrypoint"):
        target.run(tmp_path, "a" * 64, "transfer", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mode", ["train-smoke", "transfer"])
def test_inference_cli_rejects_unfrozen_comparison_before_output(mode, tmp_path):
    from scripts import evaluate_condition_relevance_transfer as target

    reference = tmp_path / "comparison"
    reference.mkdir()
    raw = json.dumps({"status": "RUNNING"}).encode()
    (reference / "result.json").write_bytes(raw)
    output = tmp_path / "inference"
    response = subprocess.run(
        [
            sys.executable,
            "-m",
            target.__name__,
            "--comparison",
            str(reference),
            "--comparison-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--mode",
            mode,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert response.returncode != 0
    assert "completed frozen comparison required" in response.stderr
    assert not output.exists()
