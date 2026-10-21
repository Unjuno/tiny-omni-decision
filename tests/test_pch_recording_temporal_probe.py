import numpy as np
import pytest

from scripts import pch_recording_temporal_probe as probe


def _order(
    length: int, mode: str, seed: int = 17, sample_id: str = "P001/recording-a"
) -> list[int]:
    function = getattr(probe, "sequence_order_indices", None)
    assert callable(function), "sequence-ordering behavior must be implemented"
    return function(length, mode=mode, seed=seed, sample_id=sample_id)


def test_chronological_mode_preserves_recording_order() -> None:
    assert _order(4, "chronological") == [0, 1, 2, 3]


def test_reverse_mode_reverses_only_within_recording() -> None:
    assert _order(4, "reverse") == [3, 2, 1, 0]


def test_shuffle_mode_is_stable_for_the_pinned_seed_and_sample() -> None:
    assert _order(6, "shuffle") == [5, 1, 2, 0, 4, 3]


def test_shuffle_mode_uses_sample_identity_to_avoid_shared_permutations() -> None:
    first = _order(10, "shuffle", sample_id="P001/recording-a")
    second = _order(10, "shuffle", sample_id="P001/recording-b")
    assert first != second
    assert sorted(first) == list(range(10))
    assert sorted(second) == list(range(10))


@pytest.mark.parametrize("length", [0, -1])
def test_sequence_order_rejects_empty_lengths(length: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        _order(length, "chronological")


def test_sequence_order_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        _order(3, "randomish")


def _required_function(name: str):
    function = getattr(probe, name, None)
    assert callable(function), f"{name} behavior must be implemented"
    return function


def test_attempt_config_uses_its_external_output_directory_name() -> None:
    config = _required_function("effective_config_for_attempt")(
        probe.Path("attempt-03")
    )
    assert config["experiment_id"] == "attempt-03"
    assert config["seed"] == probe.CONFIG["seed"]


def test_hotgpu_command_uses_absolute_python_and_concrete_paths() -> None:
    command = _required_function("hotgpu_command_text")(
        probe.Path("source.zip"),
        probe.Path("baseline"),
        probe.Path("attempt-03"),
        python=r"C:\Python311\python.exe",
    )
    assert r"C:\Python311\python.exe" in command
    assert "<Python 3.11>" not in command
    assert "attempt-03" in command
    assert "run_with_hotgpu_guard" in command


def test_flat_window_embeddings_are_split_by_exact_record_lengths() -> None:
    embeddings = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    sequences = _required_function("split_window_embeddings")(embeddings, [2, 3])
    assert [sequence.tolist() for sequence in sequences] == [
        [[0.0, 1.0], [2.0, 3.0]],
        [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
    ]


def test_flat_window_embeddings_reject_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="sum"):
        _required_function("split_window_embeddings")([[1.0], [2.0]], [1, 2])


def test_padded_batches_keep_lengths_and_zero_pad_only_the_tail() -> None:
    padded, lengths = _required_function("pad_sequence_batch")([[[1.0], [2.0]], [[3.0]]])
    assert padded.tolist() == [[[1.0], [2.0]], [[3.0], [0.0]]]
    assert lengths == [2, 1]


def test_padded_batches_reject_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        _required_function("pad_sequence_batch")([[[1.0]], []])


def test_temporal_classifier_ignores_values_after_each_declared_length() -> None:
    torch = pytest.importorskip("torch")
    model = _required_function("build_temporal_classifier")(
        torch, input_size=2, hidden_size=3, class_count=2
    )
    model.eval()
    first = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [8.0, 8.0]]])
    changed_padding = first.clone()
    changed_padding[0, 2] = torch.tensor([-9.0, 11.0])
    with torch.inference_mode():
        left = model(first, [2])
        right = model(changed_padding, [2])
    assert torch.allclose(left, right, atol=0.0, rtol=0.0)


def test_identical_probabilities_have_zero_subject_cluster_difference() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    subjects = ["participant-a", "participant-a", "participant-b", "participant-b"]
    result = _required_function("paired_subject_bootstrap")(
        probabilities, probabilities.copy(), targets, subjects, seed=17, replicates=100
    )
    assert result["delta_direction"] == "condition_a_minus_condition_b"
    assert result["accuracy_difference_95_percent_interval"] == [0.0, 0.0]
    assert result["nll_difference_95_percent_interval"] == [0.0, 0.0]


def test_subject_cluster_bootstrap_rejects_misaligned_inputs() -> None:
    probabilities = np.full((2, 2), 0.5, dtype=np.float64)
    with pytest.raises(ValueError, match="aligned"):
        _required_function("paired_subject_bootstrap")(
            probabilities, probabilities, np.asarray([0, 1]), ["participant-a"]
        )


def test_frozen_window_encoder_emits_one_vector_per_window() -> None:
    torch = pytest.importorskip("torch")
    encoder = _required_function("build_window_encoder")(torch, hidden_size=4)
    with torch.inference_mode():
        result = encoder(torch.zeros((3, 128, 3), dtype=torch.float32))
    assert result.shape == (3, 4)


def test_sequence_training_saves_best_and_final_and_reload_predictions(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    sequences = [
        np.asarray([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.5, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 2.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 1.5], [0.0, 2.0]], dtype=np.float32),
    ]
    result = _required_function("train_sequence_model")(
        torch,
        train_sequences=sequences,
        train_targets=np.asarray([0, 0, 1, 1]),
        train_sample_ids=["train-a", "train-b", "train-c", "train-d"],
        validation_sequences=sequences,
        validation_targets=np.asarray([0, 0, 1, 1]),
        validation_sample_ids=["valid-a", "valid-b", "valid-c", "valid-d"],
        order_mode="chronological",
        output_dir=tmp_path / "ordered",
        config={
            "seed": 17,
            "hidden_size": 2,
            "batch_size": 2,
            "max_epochs": 2,
            "patience": 2,
            "min_delta": 0.0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        device="cpu",
        class_count=2,
    )
    assert result["best_epoch"] in {1, 2}
    assert result["probabilities"].shape == (4, 2)
    assert np.isfinite(result["probabilities"]).all()
    assert np.allclose(result["probabilities"].sum(axis=1), 1.0)
    assert (tmp_path / "ordered" / "best.pt").is_file()
    assert (tmp_path / "ordered" / "final.pt").is_file()
    assert (tmp_path / "ordered" / "history.jsonl").is_file()


def test_selected_checkpoint_can_be_replayed_under_a_different_order(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    sequences = [
        np.asarray([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.5, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 2.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 1.5], [0.0, 2.0]], dtype=np.float32),
    ]
    config = {
        "seed": 17,
        "hidden_size": 2,
        "batch_size": 2,
        "max_epochs": 2,
        "patience": 2,
        "min_delta": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    output = tmp_path / "ordered"
    _required_function("train_sequence_model")(
        torch,
        train_sequences=sequences,
        train_targets=np.asarray([0, 0, 1, 1]),
        train_sample_ids=["train-a", "train-b", "train-c", "train-d"],
        validation_sequences=sequences,
        validation_targets=np.asarray([0, 0, 1, 1]),
        validation_sample_ids=["valid-a", "valid-b", "valid-c", "valid-d"],
        order_mode="chronological",
        output_dir=output,
        config=config,
        device="cpu",
        class_count=2,
    )
    probabilities = _required_function("predict_sequence_model")(
        torch,
        output / "best.pt",
        sequences,
        ["valid-a", "valid-b", "valid-c", "valid-d"],
        order_mode="reverse",
        config=config,
        device="cpu",
        class_count=2,
    )
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_metrics_report_class_support_and_accuracy() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]], dtype=np.float64)
    metrics = _required_function("classification_metrics")(
        probabilities, np.asarray([0, 1, 0], dtype=np.int64)
    )
    assert metrics["class_counts"] == {"0": 2, "1": 1}
    assert metrics["class_accuracy"] == {"0": 1.0, "1": 1.0}


def test_window_encoder_loads_only_the_prefixed_lstm_checkpoint_weights() -> None:
    torch = pytest.importorskip("torch")
    source = _required_function("build_window_encoder")(torch, hidden_size=4)
    checkpoint_state = {key: value.clone() for key, value in source.state_dict().items()}
    checkpoint_state["classifier.weight"] = torch.zeros((18, 4))
    checkpoint_state["classifier.bias"] = torch.zeros(18)
    loaded = _required_function("load_window_encoder_from_checkpoint")(
        torch, checkpoint_state, hidden_size=4
    )
    inputs = torch.arange(2 * 128 * 3, dtype=torch.float32).reshape(2, 128, 3) / 100.0
    with torch.inference_mode():
        expected = source(inputs)
        actual = loaded(inputs)
    assert torch.equal(actual, expected)
