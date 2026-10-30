import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from scripts.train_starss23_readouts import (  # noqa: E402
    candidate_inputs,
    paired_room_bootstrap,
    standardize,
    temperature_from_calibration,
    train_condition,
)


def test_standardization_uses_training_only():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]], dtype=np.float32)
    _, mean, std = standardize(values, np.array([True, True, False]))
    np.testing.assert_array_equal(mean, [2, 3])
    np.testing.assert_array_equal(std, [1, 1])


def test_candidate_option_identity_and_query_order():
    features = torch.zeros(2, 1152)
    result = candidate_inputs(features, torch.tensor([0, 13]))
    assert result.shape == (2, 2, 1166)
    assert torch.equal(result[0, :, -2:], torch.eye(2))
    assert result[1, 0, 1153] == 1


def test_temperature_and_bootstrap_are_deterministic():
    logits = torch.tensor([[[0.0, 2.0]] * 12, [[2.0, 0.0]] * 12])
    targets = torch.tensor([[1] * 12, [0] * 12])
    temperature, _ = temperature_from_calibration(logits, targets)
    assert temperature == 0.25
    p = np.full((2, 12), 0.5)
    result = paired_room_bootstrap(targets.numpy(), p, p, ["a", "b"], 100)
    assert all(interval == [0, 0] for interval in result["macro_intervals"].values())
    assert result["low_power"]


def test_small_synthetic_training_saves_best_final_and_predictions(tmp_path):
    torch.set_num_threads(1)
    features = np.random.default_rng(17).normal(size=(6, 1152)).astype(np.float32)
    targets = np.array([[0, 1] * 6] * 6)
    masks = {
        s: np.array([i // 2 == j for i in range(6)])
        for j, s in enumerate(("train", "calibration", "validation"))
    }
    config = {"learning_rate": 0.001, "weight_decay": 0.001, "epochs": 2, "batch_questions": 128}
    _, temperature, probabilities, result = train_condition(
        features, targets, masks, config, tmp_path
    )
    assert 0.25 <= temperature <= 4
    assert probabilities.shape == (2, 12)
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "final.pt").is_file()
    assert result["best_epoch"] in (1, 2)
