import json

import pytest

torch = pytest.importorskip("torch")


def _rows():
    return [
        {
            "image_filename": f"CLEVR_train_{i:06}.png",
            "question_index": 0,
            "question": "How many objects are there?",
            "answer": str(i % 2),
            "program": [{"function": "count"}],
        }
        for i in range(8)
    ]


def test_prepare_options_is_deterministic_and_same_kind():
    from tiny_omni_decision.clevr_vqa_probe import prepare_option_tensors

    rows = _rows()
    first = prepare_option_tensors(rows)
    second = prepare_option_tensors(rows)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    assert first[0].shape == (8, 4)
    assert first[3][0]["sample_id"] == "CLEVR_train_000000.png:0"
    assert first[3][0]["answer_kind"] == "count"
    assert all(len(row["options"]) == 4 for row in first[3])


def test_manifest_options_and_target_order_are_used_as_written():
    from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS
    from tiny_omni_decision.clevr_vqa_probe import prepare_option_tensors

    row = {
        **_rows()[0],
        "options": ["2", "0", "1", "3"],
        "target_position": 1,
    }
    candidate_indices, option_counts, targets, _ = prepare_option_tensors([row])
    assert option_counts.tolist() == [4]
    assert targets.tolist() == [1]
    assert candidate_indices[0, :4].tolist() == [
        ANSWER_LABELS.index(answer) for answer in row["options"]
    ]


def test_frozen_feature_fit_keeps_best_and_terminal_checkpoint(tmp_path):
    from tiny_omni_decision.clevr_vqa_probe import evaluate_readout, fit_readout
    from tiny_omni_decision.clevr_vqa_readout import ClevrAnswerReadout

    rows = _rows()
    features = torch.randn(8, 768, generator=torch.Generator().manual_seed(9))
    output = tmp_path / "attempt"
    result = fit_readout(
        features[:6],
        rows[:6],
        features[6:],
        rows[6:],
        output_dir=output,
        epochs=2,
        batch_size=3,
        seed=17,
        validation_controls={
            "blank_image": features[6:] * 0,
            "shuffled_image": features[6:].flip(0),
        },
    )
    assert result["epochs_completed"] == 2
    assert result["trainable_parameters"] == 28 * 768 + 28
    assert 1 <= result["best_epoch"] <= 2
    assert (output / "best-readout.pt").is_file()
    assert (output / "terminal-readout.pt").is_file()
    assert (output / "best-blank_image-predictions.jsonl").is_file()
    assert (output / "best-shuffled_image-predictions.jsonl").is_file()
    assert "blank_image" in result["history"][0]["validation_controls"]
    predictions = [
        json.loads(line) for line in (output / "best-predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 2
    assert len(predictions[0]["option_probabilities"]) == 4
    assert predictions[0]["prediction"] in predictions[0]["options"]
    assert len(json.loads((output / "history.json").read_text())) == 2

    model = ClevrAnswerReadout()
    model.load_state_dict(torch.load(output / "best-readout.pt", weights_only=True))
    metrics, reloaded_predictions = evaluate_readout(model, features[6:], rows[6:])
    assert metrics["overall"]["count"] == 2
    assert metrics["uniform_choice_baseline"]["overall"]["nll"] == pytest.approx(
        torch.log(torch.tensor(4.0)).item()
    )
    assert [row["prediction"] for row in reloaded_predictions] == [
        row["prediction"] for row in predictions
    ]


def test_fit_refuses_to_overwrite_output(tmp_path):
    from tiny_omni_decision.clevr_vqa_probe import fit_readout

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        fit_readout(
            torch.zeros(1, 768),
            _rows()[:1],
            torch.zeros(1, 768),
            _rows()[1:2],
            output_dir=output,
            epochs=1,
        )
