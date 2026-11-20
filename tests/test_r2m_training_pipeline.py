import numpy as np
import torch

from scripts import train_r2m_activity_decision as trainer

make_model = trainer.make_model
pack_batch = trainer.pack_batch


def _rows():
    return [
        {
            "sample_id": "trial-a",
            "activity": "c0",
            "duration": 2.0,
            "audio": np.ones((3, 384), dtype=np.float32),
            "sensor": np.ones((128, 3), dtype=np.float32),
        },
        {
            "sample_id": "trial-b",
            "activity": "c20",
            "duration": 1.0,
            "audio": np.zeros((2, 384), dtype=np.float32),
            "sensor": np.zeros((128, 3), dtype=np.float32),
        },
    ]


def test_training_input_condition_masks_sensor_without_changing_architecture():
    vocabulary = tuple(f"c{index}" for index in range(21))
    rows = _rows()
    normal, *_ = pack_batch(rows, vocabulary, "audio_sensor_gated")
    absent, queries, labels, option_valid = pack_batch(
        rows,
        vocabulary,
        "audio_sensor_gated",
        modality_condition="without_sensor",
    )

    assert normal["sensor"].present.tolist() == [True, True]
    assert absent["sensor"].present.tolist() == [False, False]
    assert not absent["sensor"].valid.any()
    assert absent["audio"].present.tolist() == [True, True]
    assert absent["audio"].valid.sum(dim=1).tolist() == [3, 2]

    torch.manual_seed(17)
    fusion, scorer = make_model("audio_sensor_gated", len(vocabulary), torch.device("cpu"))
    logits = scorer(fusion(absent, queries, option_valid)).squeeze(-1)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()

    parameter_count = sum(parameter.numel() for parameter in fusion.parameters())
    parameter_count += sum(parameter.numel() for parameter in scorer.parameters())
    assert parameter_count == 180_387


def test_train_arm_uses_sensor_absent_condition_for_fit_and_selection(
    tmp_path, monkeypatch
):
    vocabulary = tuple(f"c{index}" for index in range(21))
    rows = [
        {
            "sample_id": f"trial-{index}",
            "participant_id": 3,
            "trial_id": index,
            "activity": label,
            "duration": 1.0,
            "audio": np.full((2, 384), index, dtype=np.float32),
            "sensor": np.full((128, 3), index, dtype=np.float32),
        }
        for index, label in enumerate(vocabulary)
    ]
    conditions = []
    original_pack_batch = trainer.pack_batch

    def capture_condition(*args, **kwargs):
        conditions.append(kwargs.get("modality_condition", "normal"))
        return original_pack_batch(*args, **kwargs)

    monkeypatch.setattr(trainer, "pack_batch", capture_condition)
    summary, _ = trainer.train_arm(
        "audio_sensor_gated",
        rows,
        rows,
        rows,
        vocabulary,
        np.ones(len(vocabulary), dtype=np.float32),
        tmp_path,
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.1,
        device=torch.device("cpu"),
        sensor_normalization=None,
        modality_condition="without_sensor",
    )

    assert summary["modality_condition"] == "without_sensor"
    assert len(conditions) >= 4
    assert set(conditions) == {"without_sensor"}
