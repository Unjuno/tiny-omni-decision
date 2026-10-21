from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transfer = import_module("scripts.evaluate_clevrer_synthetic_av_v1_transfer_v2")
runner = import_module("scripts.train_clevrer_synthetic_av_temporal_decision")
CausalGatedTemporalDecision = import_module(
    "tiny_omni_decision.temporal_decision"
).CausalGatedTemporalDecision
PROTOCOL_SHA256_BY_EXPERIMENT = runner.PROTOCOL_SHA256_BY_EXPERIMENT


def test_transfer_evaluates_v1_lstm_checkpoints_without_mean_pooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "experiment_id": "clevrer-synthetic-av-temporal-decision-v2",
        "protocol_sha256": PROTOCOL_SHA256_BY_EXPERIMENT[
            "clevrer-synthetic-av-temporal-decision-v2"
        ],
        "readout": {"video_size": 768, "audio_size": 1, "text_size": 384, "hidden_size": 4},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    feature_report_path = tmp_path / "feature-report.json"
    feature_report_path.write_text("{}", encoding="utf-8")
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()

    validation_rows = [
        {
            "id": f"sample-{target}",
            "pair_id": "pair-1",
            "scene_group_id": "scene-1",
            "media_sha256": "media-1",
            "question": "Which came first?",
            "options": ["sound before collision", "sound after collision"],
            "target_index": target,
            "target": "sound before collision" if target == 0 else "sound after collision",
        }
        for target in (0, 1)
    ]
    tensors = {
        "video": torch.zeros(2, 16, 768),
        "audio": torch.zeros(2, 16, 1),
        "text": torch.zeros(2, 384),
        "targets": torch.tensor([0, 1]),
    }
    input_report = {
        "split": {
            "train_count": 0,
            "validation_id_order_sha256": "a" * 64,
            "sample_manifest_sha256": {"train.jsonl": "b" * 64},
        }
    }
    monkeypatch.setattr(
        transfer,
        "_load_inputs",
        lambda *_args: ([], validation_rows, tensors, input_report),
    )
    mean_pooling_flags: list[bool] = []

    def fake_logits(_model, _video, _audio, _text, *, mean_pooled: bool):
        mean_pooling_flags.append(mean_pooled)
        return torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    monkeypatch.setattr(transfer, "_model_logits", fake_logits)

    v1_config = {
        "experiment_id": transfer.V1_EXPERIMENT_ID,
        "protocol_sha256": PROTOCOL_SHA256_BY_EXPERIMENT[transfer.V1_EXPERIMENT_ID],
    }
    v1_run_dir = tmp_path / "v1"
    v1_run_dir.mkdir()
    (v1_run_dir / "run-metadata.json").write_text("{}", encoding="utf-8")
    (v1_run_dir / "results.json").write_text("{}", encoding="utf-8")
    state = CausalGatedTemporalDecision(
        video_size=768, audio_size=1, text_size=384, hidden_size=4
    ).state_dict()
    for candidate in transfer.CANDIDATES:
        torch.save(
            {
                "candidate": candidate,
                "selector": "minimum validation NLL",
                "config": v1_config,
                "step": 128,
                "model_state": state,
            },
            v1_run_dir / f"best-{candidate}.pt",
        )

    output_dir = tmp_path / "output"
    result = transfer.evaluate_transfer(
        config_path=config_path,
        feature_dir=tmp_path,
        feature_report_path=feature_report_path,
        sample_dir=sample_dir,
        v1_run_dir=v1_run_dir,
        output_dir=output_dir,
    )

    assert mean_pooling_flags == [False, False, False]
    assert result["status"] == "complete_v1_transfer_evaluation_on_v2_task_specific_validation"
    assert len(result["candidate_results"]) == 3
