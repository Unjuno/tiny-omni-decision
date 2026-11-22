from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from scripts.verify_massive_ja_readout_reload import verify_reload


def test_verify_reload_reproduces_saved_probabilities(tmp_path):
    head = nn.Sequential(nn.Linear(4, 128), nn.Tanh(), nn.Dropout(0.1), nn.Linear(128, 60))
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    targets = torch.tensor([0, 1, 2])
    with torch.inference_mode():
        probabilities = head.eval()(features).softmax(-1).tolist()
    feature_path = tmp_path / "features.pt"
    checkpoint_path = tmp_path / "head.pt"
    predictions_path = tmp_path / "predictions.jsonl"
    torch.save({"validation_features": features, "validation_targets": targets}, feature_path)
    torch.save(head.state_dict(), checkpoint_path)
    predictions_path.write_text(
        "".join(
            json.dumps({"target_index": int(target), "probabilities": probability}) + "\n"
            for target, probability in zip(targets.tolist(), probabilities, strict=True)
        ),
        encoding="utf-8",
    )

    receipt = verify_reload(feature_path, checkpoint_path, predictions_path, hidden_size=4)

    assert receipt["status"] == "pass"
    assert receipt["rows"] == 3
    assert receipt["matching_argmax_rows"] == 3
    assert receipt["max_probability_abs_delta"] == 0.0


def test_verify_reload_fails_on_target_order_mismatch(tmp_path):
    head = nn.Sequential(nn.Linear(4, 128), nn.Tanh(), nn.Dropout(0.1), nn.Linear(128, 60))
    features = torch.zeros((2, 4))
    targets = torch.tensor([0, 1])
    with torch.inference_mode():
        probabilities = head.eval()(features).softmax(-1).tolist()
    feature_path = tmp_path / "features.pt"
    checkpoint_path = tmp_path / "head.pt"
    predictions_path = tmp_path / "predictions.jsonl"
    torch.save({"validation_features": features, "validation_targets": targets}, feature_path)
    torch.save(head.state_dict(), checkpoint_path)
    predictions_path.write_text(
        "".join(
            json.dumps({"target_index": 1 - int(target), "probabilities": probability})
            + "\n"
            for target, probability in zip(targets.tolist(), probabilities, strict=True)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target order"):
        verify_reload(feature_path, checkpoint_path, predictions_path, hidden_size=4)
