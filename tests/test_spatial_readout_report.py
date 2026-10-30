import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.report_spatial_trajectory_readout import audit_epoch  # noqa: E402
from scripts.train_role_supervised_qa import qa_summary  # noqa: E402


def test_publication_json_has_stable_lf_bytes(tmp_path):
    import json

    from scripts.report_spatial_trajectory_readout import write_publication

    path = tmp_path / "report.json"
    write_publication(path, {"status": "記録", "nll": 1.0})
    assert b"\r\n" not in path.read_bytes()
    assert json.loads(path.read_bytes()) == {"status": "記録", "nll": 1.0}


def test_saved_labels_bind_original_ids_targets_and_optional_cover():
    from scripts.report_spatial_trajectory_readout import verify_original_labels

    original = dict(
        qa=np.array([0, 2]),
        identities=np.array(["video_1"]),
        cover=np.array([[True, False]]),
        available=np.array([True]),
    )
    verify_original_labels(original, original)
    for key in original:
        changed = {k: v.copy() for k, v in original.items()}
        if key == "identities":
            changed[key][0] = "video_2"
        else:
            changed[key].flat[0] = 0 if changed[key].flat[0] else 1
        with pytest.raises(ValueError, match="original"):
            verify_original_labels(changed, original)


def test_original_question_metrics_recomputed_and_corruption_rejected():
    targets = np.array([0, 2])
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    p, logp = logits.softmax(-1).numpy(), logits.log_softmax(-1).numpy()
    saved = dict(valid=p, valid_logp=logp)
    recorded = dict(valid=qa_summary(p, targets, logp))
    audit_epoch(saved, dict(valid=targets), recorded)
    recorded["valid"]["initial"]["accuracy"] += 0.01
    with pytest.raises(ValueError, match="metrics"):
        audit_epoch(saved, dict(valid=targets), recorded)
    recorded = dict(valid=qa_summary(p, targets, logp))
    recorded["valid"]["nll"] += 0.01
    with pytest.raises(ValueError, match="metrics"):
        audit_epoch(saved, dict(valid=targets), recorded)
    with pytest.raises(ValueError, match="denominator"):
        audit_epoch(dict(valid=p[:1], valid_logp=logp[:1]), dict(valid=targets), recorded)
    with pytest.raises(ValueError, match="inconsistency"):
        audit_epoch(dict(valid=p, valid_logp=logp + 1), dict(valid=targets), recorded)
