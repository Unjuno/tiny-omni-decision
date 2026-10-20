from __future__ import annotations

import json
from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
temporal_runner = import_module("scripts.train_clevrer_synthetic_av_temporal_decision")
_metrics = temporal_runner._metrics
_run_candidate = temporal_runner._run_candidate
_pair_donor_indices = temporal_runner._pair_donor_indices


def _rows(split: str, scene: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"{split}-{scene}-before",
            "pair_id": f"{split}-{scene}-event",
            "scene_group_id": f"clevrer:{split}:{scene}",
            "media_sha256": f"media-{split}-{scene}",
            "question": "Did the sound happen before or after the collision?",
            "options": ["sound before collision", "sound after collision"],
            "target_index": target,
            "target": "sound before collision" if target == 0 else "sound after collision",
        }
        for target in (0, 1)
    ]


def test_metrics_report_accuracy_nll_brier_and_calibration() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    metrics = _metrics(logits, torch.tensor([0, 1]))

    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] < 0.1
    assert metrics["brier"] < 0.02
    assert metrics["ece_15_bins"] < 0.1


def test_audio_intervention_swaps_only_within_the_same_paired_event() -> None:
    rows = _rows("validation", 3) + _rows("validation", 4)

    donors = _pair_donor_indices(rows).tolist()

    assert donors == [1, 0, 3, 2]
    assert all(
        rows[index]["target_index"] != rows[donors[index]]["target_index"] for index in range(4)
    )


def test_matched_runner_preserves_best_and_final_checkpoints(tmp_path) -> None:
    torch.manual_seed(17)
    rows_train = _rows("train", 1) + _rows("train", 2)
    rows_validation = _rows("validation", 3) + _rows("validation", 4)
    video = torch.randn(8, 16, 768)
    audio = torch.zeros(8, 16, 1)
    audio[0, 4, 0] = 1.0
    audio[1, 8, 0] = 1.0
    audio[2, 5, 0] = 1.0
    audio[3, 9, 0] = 1.0
    audio[4, 6, 0] = 1.0
    audio[5, 10, 0] = 1.0
    audio[6, 7, 0] = 1.0
    audio[7, 11, 0] = 1.0
    text = torch.randn(8, 384)
    text[1] = text[0]
    text[3] = text[2]
    text[5] = text[4]
    text[7] = text[6]
    video[1] = video[0]
    video[3] = video[2]
    video[5] = video[4]
    video[7] = video[6]
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    config = {
        "seed": 17,
        "training": {
            "max_optimizer_updates_per_candidate": 4,
            "evaluation_interval": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "readout": {
            "video_size": 768,
            "audio_size": 1,
            "text_size": 384,
            "hidden_size": 4,
        },
    }
    orders = [torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])]

    result = _run_candidate(
        name="joint_mean_pooled",
        mode="joint",
        mean_pooled=True,
        config=config,
        train_rows=rows_train,
        validation_rows=rows_validation,
        tensors={"video": video, "audio": audio, "text": text, "targets": targets},
        train_count=4,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        epoch_orders=orders,
    )

    assert result["optimizer_updates"] == 4
    assert result["consumed_training_examples"] == 8
    assert result["unique_training_examples_consumed"] == 4
    assert result["unique_training_scenes_consumed"] == 2
    assert result["unique_training_media_assets_consumed"] == 2
    assert result["best_step"] in {0, 2, 4}
    assert (tmp_path / "best-joint_mean_pooled.pt").is_file()
    assert (tmp_path / "final-joint_mean_pooled.pt").is_file()
    assert (
        len((tmp_path / "training-history-joint_mean_pooled.jsonl").read_text().splitlines()) == 3
    )
    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions-best-joint_mean_pooled.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(predictions) == 4
    assert all(len(row["probabilities"]) == 2 for row in predictions)
