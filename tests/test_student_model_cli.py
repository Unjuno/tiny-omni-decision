import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def test_model_module_exists():
    assert importlib.util.find_spec("tiny_omni_decision.student.model") is not None


def make_record():
    from tiny_omni_decision.student.data import Record
    return Record("id", "group", "content", "text", {"text": "question"},
                  ["first", "second", "third"], 1, [0.0, 1.0, 0.0], {})


class TinyEncoder(torch.nn.Module):
    """Real differentiable fixture for the documented preprocess/forward interface."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 5)
    def preprocess(self, inputs, **kwargs):
        values = []
        for item in inputs:
            text = item["text"] if isinstance(item, dict) else item
            values.append([sum(text.encode()) % 16, len(text) % 16])
        return {"input_ids": torch.tensor(values)}
    def forward(self, features):
        return {"sentence_embedding": self.embedding(features["input_ids"]).mean(dim=1)}
    def encode(self, *args, **kwargs):
        raise AssertionError("inference encode must not be used for training")


def test_native_readout_has_gradients_and_option_permutation_equivariance(tmp_path):
    from dataclasses import replace

    from tiny_omni_decision.student.model import EmbeddingDecisionStudent
    torch.manual_seed(17)
    student = EmbeddingDecisionStudent(TinyEncoder(), tmp_path, max_length=32,
                                       score_temperature=0.1)
    record = make_record()
    logits = student(record)
    assert logits.shape == (3,) and logits.requires_grad
    logits.square().sum().backward()
    assert student.encoder.embedding.weight.grad is not None
    swapped = replace(record, options=[record.options[i] for i in [2, 0, 1]])
    assert torch.allclose(student(swapped), logits[[2, 0, 1]])


def test_media_escape_and_hash_mismatch_refused(tmp_path):
    from tiny_omni_decision.student.data import Cache, Record, media_path, validate_media
    with pytest.raises(ValueError):
        media_path(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError):
        media_path(tmp_path, "https://example.org/track.jpg")
    (tmp_path / "pic.png").write_bytes(b"test")
    r = Record("x", "g", "c", "image", {"text": "q", "image": "pic.png"},
               ["a", "b"], 0, [1.0, 0.0], {"pic.png": "0" * 64})
    with pytest.raises(ValueError, match="checksum"):
        validate_media(Cache("train", "t", [r], "a"*64), tmp_path)


def test_sequence_overflow_refused(tmp_path):
    from tiny_omni_decision.student.model import EmbeddingDecisionStudent
    student = EmbeddingDecisionStudent(TinyEncoder(), tmp_path, max_length=1,
                                       score_temperature=0.1)
    with pytest.raises(ValueError, match="sequence"):
        student(make_record())


def test_pin_must_be_immutable_before_loading_dependencies():
    from tiny_omni_decision.student.model import validate_pin
    with pytest.raises(ValueError, match="revision"):
        validate_pin({"model_id": "google/embeddinggemma-2", "revision": "main"})
    validate_pin({"model_id": "google/embeddinggemma-2", "revision": "a"*40,
                  "model_type": "embedding_gemma2", "license": "apache-2.0"})


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
    values = json.loads((root / "configs/student/ternary_first.example.json").read_text())
    validate_config(values)
    with pytest.raises(ValueError):
        validate_config(values | {"qat_steps": 0})
