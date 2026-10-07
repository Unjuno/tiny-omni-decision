import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_model_module_exists():
    assert importlib.util.find_spec("tiny_omni_decision.student.model") is not None


def make_record():
    from tiny_omni_decision.student.data import Record

    return Record(
        "id",
        "group",
        "content",
        "text",
        {"text": "question"},
        ["first", "second", "third"],
        1,
        [0.0, 1.0, 0.0],
        {},
    )


class TinyEncoder(torch.nn.Module):
    """Differentiable fixture exposing token embeddings for custom pooling."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 5)

    def get_sentence_embedding_dimension(self):
        return 5

    def preprocess(self, inputs, **kwargs):
        values = []
        for item in inputs:
            text = item["text"] if isinstance(item, dict) else item
            values.append([sum(text.encode()) % 16, len(text) % 16, 0])
        return {
            "input_ids": torch.tensor(values),
            "attention_mask": torch.tensor([[1, 1, 0]] * len(values)),
        }

    def forward(self, features):
        token_embeddings = self.embedding(features["input_ids"])
        return {
            "token_embeddings": token_embeddings,
            "attention_mask": features["attention_mask"],
            # Deliberately bogus: the student must replace native sentence pooling.
            "sentence_embedding": torch.full(
                (token_embeddings.shape[0], token_embeddings.shape[-1]),
                123.0,
                device=token_embeddings.device,
            ),
        }

    def encode(self, *args, **kwargs):
        raise AssertionError("inference encode must not be used for training")


def test_custom_mean_pool_ignores_masked_tokens():
    from tiny_omni_decision.student.model import mean_pool

    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(tokens, mask)
    assert torch.equal(pooled, torch.tensor([[2.0, 3.0]]))


def test_tiny_decision_head_has_gradients_and_option_permutation_equivariance(tmp_path):
    from dataclasses import replace

    from tiny_omni_decision.student.model import EmbeddingDecisionStudent

    torch.manual_seed(17)
    student = EmbeddingDecisionStudent(
        TinyEncoder(), tmp_path, max_length=32, head_hidden_dim=7
    )
    record = make_record()
    logits = student(record)
    assert logits.shape == (3,) and logits.requires_grad
    logits.square().sum().backward()
    assert student.encoder.embedding.weight.grad is not None
    assert any(p.grad is not None for p in student.decision_head.parameters())

    swapped = replace(record, options=[record.options[i] for i in [2, 0, 1]])
    assert torch.allclose(student(swapped), logits[[2, 0, 1]], atol=1e-6)


def test_decision_head_is_explicit_high_precision_qat_exception(tmp_path):
    from tiny_omni_decision.student.model import EmbeddingDecisionStudent
    from tiny_omni_decision.student.ternary import TernaryController

    student = EmbeddingDecisionStudent(
        TinyEncoder(), tmp_path, max_length=32, head_hidden_dim=7
    )
    exclusions = student.ternary_exclusions()
    assert exclusions
    controller = TernaryController(student, group_size=4, exclude=exclusions)
    student.enable_decision_head_training()
    assert not any(name in controller.targets for name in exclusions)
    assert all(p.requires_grad for p in student.decision_head.parameters())
    assert any(name.startswith("encoder.") for name in controller.targets)


def test_media_escape_and_hash_mismatch_refused(tmp_path):
    from tiny_omni_decision.student.data import Cache, Record, media_path, validate_media

    with pytest.raises(ValueError):
        media_path(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError):
        media_path(tmp_path, "https://example.org/track.jpg")
    (tmp_path / "pic.png").write_bytes(b"test")
    r = Record(
        "x",
        "g",
        "c",
        "image",
        {"text": "q", "image": "pic.png"},
        ["a", "b"],
        0,
        [1.0, 0.0],
        {"pic.png": "0" * 64},
    )
    with pytest.raises(ValueError, match="checksum"):
        validate_media(Cache("train", "t", [r], "a" * 64), tmp_path)


def test_sequence_overflow_refused(tmp_path):
    from tiny_omni_decision.student.model import EmbeddingDecisionStudent

    student = EmbeddingDecisionStudent(
        TinyEncoder(), tmp_path, max_length=1, head_hidden_dim=7
    )
    with pytest.raises(ValueError, match="sequence"):
        student(make_record())


def test_pin_must_be_immutable_before_loading_dependencies():
    from tiny_omni_decision.student.model import validate_pin

    with pytest.raises(ValueError, match="revision"):
        validate_pin({"model_id": "google/embeddinggemma-2", "revision": "main"})
    validate_pin(
        {
            "model_id": "google/embeddinggemma-2",
            "revision": "a" * 40,
            "model_type": "embedding_gemma2",
            "license": "apache-2.0",
        }
    )


def test_cli_offline_smoke(tmp_path, capsys):
    from tiny_omni_decision.student.__main__ import main

    output = tmp_path / "smoke"
    assert main(["smoke", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "synthetic_pipeline_check_not_model_quality"
    assert result["lora_used"]
    assert result["reload_max_abs_error"] == 0.0
    assert (output / "result.json").is_file()


def test_example_config_is_valid():
    from tiny_omni_decision.student.__main__ import validate_config

    root = Path(__file__).parents[1]
    values = json.loads(
        (root / "configs/student/ternary_first.example.json").read_text()
    )
    validate_config(values)
    with pytest.raises(ValueError):
        validate_config(values | {"qat_steps": 0})


def test_student_requirements_match_embeddinggemma2_runtime_contract():
    root = Path(__file__).parents[1]
    requirements = (root / "requirements-student.txt").read_text(encoding="utf-8")
    assert "sentence-transformers[image,audio,video]>=6.1,<7" in requirements
    assert "transformers>=5.19,<6" in requirements


def test_embeddinggemma2_classes_exist_when_student_dependencies_are_installed():
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("student runtime dependencies are installed only in student CI")
    from transformers import EmbeddingGemma2Model, EmbeddingGemma2Processor

    assert EmbeddingGemma2Model is not None
    assert EmbeddingGemma2Processor is not None


def test_custom_pooling_bypasses_sentence_transformer_pooling_modules(tmp_path):
    from tiny_omni_decision.student.model import EmbeddingDecisionStudent

    class SentenceTransformerLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = TinyEncoder()

        def get_sentence_embedding_dimension(self):
            return self.backbone.get_sentence_embedding_dimension()

        def preprocess(self, inputs, **kwargs):
            return self.backbone.preprocess(inputs, **kwargs)

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.backbone

        def forward(self, features):
            raise AssertionError("native SentenceTransformer pooling must be bypassed")

    student = EmbeddingDecisionStudent(
        SentenceTransformerLike(), tmp_path, max_length=32, head_hidden_dim=7
    )
    logits = student(make_record())
    assert logits.shape == (3,)
