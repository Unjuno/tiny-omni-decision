import json

import pytest

pytest.importorskip("torch")
from scripts.evaluate_slot_readout import load_valid  # noqa: E402


def test_artifact_root_is_original_reuse_tree():
    from scripts.evaluate_slot_readout import ROOT

    assert ROOT.as_posix() == "C:/CodexArtifacts/pretrained-reuse-r2"


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


def test_fixed_evaluation_preserves_weights_and_saves_all_controls(tmp_path):
    import torch

    from scripts.audit_correspondence_validation import sha
    from scripts.evaluate_slot_readout import evaluate
    from tiny_omni_decision.trajectory_readout import TrajectoryReadout

    torch.manual_seed(17)
    checkpoint = tmp_path / "head.pt"
    torch.save(TrajectoryReadout(input_dim=129).state_dict(), checkpoint)
    before = sha(checkpoint)
    bank = {key: torch.randn(224, 129) for key in ["a", "b"]}
    data = dict(pairs=[dict(video_id=k, options=["a", "b", "c"]) for k in bank],
                text=torch.randn(4, 3, 1536), targets=torch.tensor([0, 1, 2, 2]))
    output = tmp_path / "evaluation"
    evaluate(bank, data, output, {"test": "synthetic"},
             {"final": (checkpoint, before), "untrained": (checkpoint, before)})
    result = json.loads((output / "result.json").read_text())
    assert result["weights_unchanged"]
    assert result["metrics"]["final"]["normal"]["count"] == 4
    assert result["bootstrap"]["videos"] == 2
    predictions = json.loads((output / "predictions.json").read_text())
    assert set(predictions["final"]) == {"normal", "mismatch", "zero"}
    assert len(predictions["final"]["normal"]) == 4
    assert sha(checkpoint) == before
    with pytest.raises(FileExistsError):
        evaluate(bank, data, output, {}, {"final": (checkpoint, before)})
