"""Optional CPU tensor checks; no pretrained model, media, GPU or training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_egoviz_temporal_controls import CONTROLS, diagnose


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def diagnostic_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    result_dir = tmp_path / "selected"
    result_dir.mkdir()
    rows = [
        {
            "sample_id": f"sample-{i}",
            "split": "train" if i < 2 else "development",
            "group_id": f"group-{i // 2}",
            "query_index": i % 4,
            "candidate_window_indices": [0, 1, 2, 3],
            "target_option_index": i % 4,
        }
        for i in range(6)
    ]
    for split in ("train", "development"):
        (corpus / f"{split}-samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["split"] == split),
            encoding="utf-8",
        )
    video_path = tmp_path / "video.npz"
    sensor_path = tmp_path / "sensor.npz"
    np.savez(
        video_path,
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        features=rng.normal(size=(6, 8, 768)).astype(np.float32),
    )
    np.savez(
        sensor_path,
        window_index=np.arange(4),
        features=rng.normal(size=(4, 6, 16, 512)).astype(np.float32),
    )
    model = JointTemporalScorer().eval()
    checkpoint = result_dir / "joint-temporal-state.pt"
    torch.save(model.state_dict(), checkpoint)
    loaded_rows, video, sensor = _load_samples(corpus, video_path, sensor_path)
    v, c, targets, groups, sample_ids, _ = _tensor_data(
        loaded_rows, video, sensor, "development"
    )
    with torch.inference_mode():
        probs = torch.softmax(model(v, c), dim=1).numpy()
    (result_dir / "development-predictions.jsonl").write_text(
        "".join(
            json.dumps({
                "sample_id": key,
                "group_id": str(groups[i]),
                "target": int(targets[i]),
                "joint_probabilities": probs[i].tolist(),
            }) + "\n"
            for i, key in enumerate(sample_ids)
        ), encoding="utf-8",
    )
    selected = {
        "sealed_audit_accessed": False,
        "input_sha256": {
            "train_samples": _hash(corpus / "train-samples.jsonl"),
            "development_samples": _hash(corpus / "development-samples.jsonl"),
            "video_features": _hash(video_path),
            "sensor_features": _hash(sensor_path),
        },
        "joint_temporal": {"state_sha256": _hash(checkpoint)},
    }
    (result_dir / "readout-results.json").write_text(json.dumps(selected), encoding="utf-8")
    try:
        yield corpus, video_path, sensor_path, result_dir, tmp_path / "diagnostic"
    finally:
        torch.set_num_threads(previous_threads)


def test_frozen_cpu_diagnostic_preserves_weights_and_reports_all_controls(diagnostic_fixture):
    corpus, video, sensor, selected, output = diagnostic_fixture
    before = _hash(selected / "joint-temporal-state.pt")
    result = diagnose(corpus, video, sensor, selected, output)
    assert list(result["controls"]) == list(CONTROLS)
    assert result["sample_count"] == 4
    assert result["development_groups"] == 2
    assert result["baseline_probability_max_abs_difference"] <= 1e-6
    assert result["training"] is False
    assert result["sealed_audit_accessed"] is False
    assert _hash(selected / "joint-temporal-state.pt") == before
    for name in ("missing_video_features", "missing_sensor_features"):
        control = result["controls"][name]
        assert control["all_options_tied_fraction"] == 1
        assert control["uniform_expected_random_tie_metrics"]["accuracy"] == 0.25
    assert result["controls"]["reverse_both_features"][
        "max_probability_difference_from_unaltered"
    ] <= 1e-6
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 7 * 4


def test_diagnostic_rejects_changed_inputs_and_duplicate_predictions(diagnostic_fixture):
    corpus, video, sensor, selected, output = diagnostic_fixture
    predictions = selected / "development-predictions.jsonl"
    original = predictions.read_text()
    predictions.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="IDs/targets differ"):
        diagnose(corpus, video, sensor, selected, output)
    assert not output.exists()
    predictions.write_text(original)
    with (corpus / "development-samples.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="inputs differ"):
        diagnose(corpus, video, sensor, selected, output)


def test_readout_report_records_actual_development_group_count(diagnostic_fixture, monkeypatch):
    import torch

    from scripts import train_egoviz_vs_sync_readout as runner

    corpus, video, sensor, _, output = diagnostic_fixture

    def frozen_fixture_fit(model, train_data, validation_data, *, joint, **kwargs):
        model.eval()
        with torch.inference_mode():
            def predict(data):
                return (model(data[0], data[1]) if joint else model(data[1])).numpy()

            return {
                "best_epoch": 1, "stopped_epoch": 1, "training_seconds": 0,
                "train_logits": predict(train_data),
                "validation_logits": predict(validation_data),
                "history": [],
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "best_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }

    monkeypatch.setattr(runner, "_fit", frozen_fixture_fit)
    result = runner.run(corpus, video, sensor, output)
    assert result["development_groups"] == 2
    assert result["limitations"][0] == (
        "development has 2 publisher session groups; participant independence is unknown"
    )
