from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def readout():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student import mean_pool_projected_tokens, supplied_option_logits

    return torch, mean_pool_projected_tokens, supplied_option_logits


def test_mean_pool_matches_attention_mask_and_keeps_prompt_tokens(readout) -> None:
    torch, mean_pool_projected_tokens, _ = readout
    embeddings = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool_projected_tokens(embeddings, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


def test_processor_tensors_move_to_model_device_without_dtype_changes(readout) -> None:
    torch, _, _ = readout
    from tiny_omni_decision.student import _move_tensor_inputs

    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "pixel_values": torch.tensor([[0.5]], dtype=torch.float32),
        "metadata": "kept",
    }

    moved = _move_tensor_inputs(inputs, device=torch.device("meta"))

    assert moved["input_ids"].device.type == "meta"
    assert moved["input_ids"].dtype == torch.int64
    assert moved["pixel_values"].device.type == "meta"
    assert moved["pixel_values"].dtype == torch.float32
    assert moved["metadata"] == "kept"


def test_decision_query_and_options_use_the_same_similarity_prefix() -> None:
    from tiny_omni_decision.student import decision_option_text, decision_query_text

    query = decision_query_text(
        "A red cube is left of a sphere.", "Which is farther?", media_token="<|video|>"
    )
    option = decision_option_text("The cube.")

    assert query == (
        "task: sentence similarity | query: A red cube is left of a sphere.\n"
        "Which is farther?\n<|video|>"
    )
    assert option == "task: sentence similarity | query: The cube."
    assert "The cube" not in query


def test_decision_example_and_options_use_processor_in_source_order(
    readout, tmp_path: Path
) -> None:
    from tiny_omni_decision.schema import DecisionExample
    from tiny_omni_decision.student import (
        processor_inputs_for_decision_example,
        processor_inputs_for_options,
    )

    class RecordingProcessor:
        image_token = "<|image|>"
        audio_token = "<|audio|>"
        video_token = "<|video|>"

        def __init__(self) -> None:
            self.calls = []

        def __call__(self, *, text, return_tensors, **payload):
            self.calls.append((text, return_tensors, payload))
            return {"text": text, **payload}

    example = DecisionExample.model_validate(
        {
            "id": "sample-1",
            "modality": "text",
            "state": "Scene details",
            "question": "Which object moved?",
            "options": ["The cube", "The sphere"],
            "target": "The cube",
            "source": "fixture",
            "source_revision": "a" * 40,
            "source_record_id": "1",
            "split": "train",
            "provenance": {"license": "CC0-1.0"},
        }
    )
    processor = RecordingProcessor()

    processor_inputs_for_decision_example(processor, example, data_root=tmp_path)
    processor_inputs_for_options(processor, example.options)

    assert processor.calls[0][0] == [
        "task: sentence similarity | query: Scene details\nWhich object moved?"
    ]
    assert processor.calls[1][0] == [
        "task: sentence similarity | query: The cube",
        "task: sentence similarity | query: The sphere",
    ]


def test_audio_example_reads_pcm_frames_while_wave_file_is_open(
    readout, tmp_path: Path
) -> None:
    import wave

    from tiny_omni_decision.schema import DecisionExample
    from tiny_omni_decision.student import processor_inputs_for_decision_example

    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16_000)
        audio_file.writeframes(b"\x01\x00" * 32)

    example = DecisionExample.model_validate(
        {
            "id": "audio-sample",
            "modality": "audio",
            "state": "",
            "question": "Which word was spoken?",
            "options": ["left", "right"],
            "target": "left",
            "media": [{"kind": "audio", "path": "sample.wav"}],
            "source": "fixture",
            "source_revision": "a" * 40,
            "source_record_id": "sample.wav",
            "split": "train",
            "provenance": {"license": "CC0-1.0"},
        }
    )

    class RecordingProcessor:
        audio_token = "<|audio|>"

        def __call__(self, *, text, return_tensors, **payload):
            return {"text": text, **payload}

    inputs = processor_inputs_for_decision_example(
        RecordingProcessor(), example, data_root=tmp_path
    )

    assert inputs["audio"][0].shape == (32,)
    assert inputs["audio"][0].dtype.name == "float32"


def test_supplied_option_logits_preserve_order_and_backpropagate(readout) -> None:
    torch, _, supplied_option_logits = readout
    query = torch.tensor([1.0, 0.0], requires_grad=True)
    options = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True
    )

    logits = supplied_option_logits(query, options, temperature=0.5)
    probabilities = torch.softmax(logits, dim=-1)
    loss = -torch.log(probabilities[1])
    loss.backward()

    torch.testing.assert_close(logits, torch.tensor([0.0, 2.0, -2.0]))
    assert probabilities.argmax().item() == 1
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert options.grad is not None and torch.isfinite(options.grad).all()
    assert query.grad.abs().sum().item() > 0
    assert options.grad.abs().sum().item() > 0


def test_encoder_forward_native_pooling_and_option_loss_backpropagate(readout) -> None:
    torch, _, supplied_option_logits = readout
    from tiny_omni_decision.student import model_sentence_embeddings

    class TinyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    model = TinyEncoder()
    query = model_sentence_embeddings(
        model,
        {"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])},
    )[0]
    options = model_sentence_embeddings(
        model,
        {
            "input_ids": torch.tensor([[3, 0], [4, 0]]),
            "attention_mask": torch.tensor([[1, 0], [1, 0]]),
        },
    )
    logits = supplied_option_logits(query, options, temperature=1.0)
    torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([0])).backward()

    assert torch.isfinite(logits).all()
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.abs().sum().item() > 0


def test_mean_pool_rejects_empty_attention_mask(readout) -> None:
    torch, mean_pool_projected_tokens, _ = readout
    with pytest.raises(ValueError, match="at least one attended token"):
        mean_pool_projected_tokens(torch.ones(1, 2, 3), torch.zeros(1, 2))


def test_option_readout_rejects_invalid_temperature_and_dimensions(readout) -> None:
    torch, _, supplied_option_logits = readout
    query = torch.ones(3)
    options = torch.ones(2, 3)
    with pytest.raises(ValueError, match="temperature"):
        supplied_option_logits(query, options, temperature=0)
    with pytest.raises(ValueError, match="dimensions"):
        supplied_option_logits(query, torch.ones(2, 4), temperature=1.0)


def test_option_readout_rejects_zero_embeddings(readout) -> None:
    torch, _, supplied_option_logits = readout
    with pytest.raises(ValueError, match="nonzero"):
        supplied_option_logits(torch.zeros(3), torch.ones(2, 3), temperature=1.0)
