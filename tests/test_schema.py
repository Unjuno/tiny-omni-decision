from pathlib import Path

from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import BaseModelManifest, DatasetManifest

ROOT = Path(__file__).resolve().parents[1]


def test_base_model_example_is_valid() -> None:
    data = load_structured_file(ROOT / "manifests" / "base-model.example.yaml")
    manifest = BaseModelManifest.model_validate(data)
    assert manifest.repo_id == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
    assert "audio" in manifest.modalities


def test_dataset_example_is_valid() -> None:
    data = load_structured_file(ROOT / "manifests" / "dataset.example.yaml")
    manifest = DatasetManifest.model_validate(data)
    assert manifest.usage == "training"
