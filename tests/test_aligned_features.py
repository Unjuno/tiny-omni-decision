import hashlib

import numpy as np
import pytest

from tiny_omni_decision.aligned_features import gather_aligned_features
from tiny_omni_decision.observations import ObservationBatch


def batch_for(tmp_path, *, times=(10., 20., 30.), valid=(True, False, True)):
    path = tmp_path / "features.npz"
    np.savez(path, values=np.array([[1., 2.], [np.nan, np.nan], [99., 98.]]),
             timestamps_ms=np.array(times), valid_mask=np.array(valid))
    batch = ObservationBatch.model_validate(dict(schema_version="1.0", observations=[dict(
        modality="video", source_id="camera", payload_ref=str(path),
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), clock_id="verified",
        timestamps_ms=times, valid_mask=valid)],
        options=[dict(option_id="a", attributes={}), dict(option_id="b", attributes={})]))
    return batch, path


def test_causal_gather_missing_mask_and_ages(tmp_path):
    batch, _ = batch_for(tmp_path)
    result = gather_aligned_features(batch, {"video": "camera", "audio": "mic"},
                                     {"video": 2, "audio": 3}, [9, 10, 20, 26, 30], max_age_ms=15)
    np.testing.assert_array_equal(result["available"]["video"], [False, True, True, False, True])
    np.testing.assert_array_equal(result["features"]["video"],
                                  [[0, 0], [1, 2], [1, 2], [0, 0], [99, 98]])
    assert not result["available"]["audio"].any()
    assert not result["features"]["audio"].any()
    assert result["selected_indices"]["video"] == (None, 0, 0, None, 2)


def test_hash_and_metadata_are_verified(tmp_path):
    batch, path = batch_for(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        gather_aligned_features(batch, {"video": "camera"}, {"video": 2}, [10], max_age_ms=15)
    batch, _ = batch_for(tmp_path)
    batch.observations[0].timestamps_ms = (11., 21., 31.)
    with pytest.raises(ValueError, match="timestamps"):
        gather_aligned_features(batch, {"video": "camera"}, {"video": 2}, [11], max_age_ms=15)


def test_wrong_modality_or_duplicate_source_rejected(tmp_path):
    batch, _ = batch_for(tmp_path)
    with pytest.raises(ValueError, match="modality"):
        gather_aligned_features(batch, {"audio": "camera"}, {"audio": 2}, [10], max_age_ms=15)
    with pytest.raises(ValueError, match="distinct"):
        gather_aligned_features(batch, {"video": "camera", "audio": "camera"},
                                {"video": 2, "audio": 2}, [10], max_age_ms=15)


def test_future_feature_change_cannot_change_earlier_gather(tmp_path):
    batch, path = batch_for(tmp_path)
    before = gather_aligned_features(batch, {"video": "camera"}, {"video": 2},
                                     [10, 20], max_age_ms=15)
    with np.load(path) as cache:
        values, stamps, valid = cache["values"], cache["timestamps_ms"], cache["valid_mask"]
    values[2] = [-999, 777]
    np.savez(path, values=values, timestamps_ms=stamps, valid_mask=valid)
    batch.observations[0].payload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    after = gather_aligned_features(batch, {"video": "camera"}, {"video": 2},
                                    [10, 20], max_age_ms=15)
    np.testing.assert_array_equal(before["features"]["video"], after["features"]["video"])


def test_gather_fusion_and_json_skip_inference_without_required_evidence(tmp_path):
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.evidence_decision import evidence_gated_json
    from tiny_omni_decision.fusion_decision import FusionDecisionScorer

    batch, _ = batch_for(tmp_path)
    gathered = gather_aligned_features(batch, {"video": "camera"}, {"video": 2},
                                       [9, 10, 26], max_age_ms=15)
    torch.manual_seed(17)
    model = FusionDecisionScorer({"video": 2}, 2, 1, width=4).eval()
    calls = []

    def predict(row):
        calls.append(row)
        with torch.inference_mode():
            logits = model(
                {"video": torch.from_numpy(gathered["features"]["video"][row:row + 1])},
                {"video": torch.from_numpy(gathered["available"]["video"][row:row + 1])},
                torch.eye(2)[None], torch.ones(1, 2, dtype=torch.bool))
        return logits.softmax(-1)[0].tolist()

    outputs = [evidence_gated_json(
        lambda row=row: predict(row), ["a", "b"],
        [{"option_id": name, "class_id": name} for name in ["a", "b"]],
        required=["video"], available={"video": bool(gathered["available"]["video"][row])})
        for row in range(3)]
    assert calls == [1]
    assert [o["status"] for o in outputs] == ["abstain", "selected", "abstain"]
