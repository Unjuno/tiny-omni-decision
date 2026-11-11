from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts import train_frozen_videoprism_clevr4_probe as clevr4_module  # noqa: E402
from scripts.cohort_gate import CohortUseBlocked  # noqa: E402
from scripts.train_frozen_videoprism_clevr4_probe import (  # noqa: E402
    _features_by_id,
    _quality_metrics,
    _tasks,
)
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES  # noqa: E402


def test_feature_cache_rejects_cross_split_duplicate_ids_and_media(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        train_ids=np.asarray(["train-1"]),
        train_media_sha256=np.asarray(["same-hash"]),
        train_features=np.zeros((1, 768), dtype=np.float32),
        validation_ids=np.asarray(["val-1"]),
        validation_media_sha256=np.asarray(["same-hash"]),
        validation_features=np.zeros((1, 768), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="train/validation identity overlap"):
        _features_by_id(path)


def test_clevr4_tasks_keep_each_taxonomy_as_a_variable_choice() -> None:
    option_embeddings = {
        f"{taxonomy}|{option}": torch.zeros(384)
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    }
    row = {
        "id": "CLEVR_new_000001",
        "color": "pink",
        "texture": "wave",
        "count": 9,
        "shape": "cylinder",
    }
    tasks, targets, image_ids, taxonomies = _tasks(
        [row], {row["id"]: torch.zeros(768)}, option_embeddings
    )
    assert len(tasks) == 4
    assert [task.shape for task in tasks] == [(10, 1152)] * 4
    assert targets == [
        CLEVR4_TAXONOMIES[taxonomy].index(str(row[taxonomy]))
        for taxonomy in CLEVR4_TAXONOMIES
    ]
    assert image_ids == [row["id"]] * 4
    assert taxonomies == list(CLEVR4_TAXONOMIES)


def test_quality_metrics_include_mean_option_confidence() -> None:
    logits = torch.tensor([2.0, 0.0])
    result = _quality_metrics([logits], [0])
    assert result["accuracy"] == 1.0
    assert result["mean_confidence"] == pytest.approx(torch.softmax(logits, dim=0).max().item())


def test_main_blocks_unknown_cohort_before_cache_or_encoder_load(tmp_path, monkeypatch) -> None:
    sample_dir = tmp_path / "sample"
    selection_dir = tmp_path / "selection"
    sample_dir.mkdir()
    selection_dir.mkdir()
    (sample_dir / "train-labels.jsonl").write_text(
        json.dumps({"id": "train-image"}) + "\n", encoding="utf-8"
    )
    (sample_dir / "val-labels.jsonl").write_text(
        json.dumps({"id": "validation-image"}) + "\n", encoding="utf-8"
    )
    (selection_dir / "val-image-manifest.jsonl").write_text(
        json.dumps({"id": "selection-image", "sha256": "a" * 64}) + "\n",
        encoding="utf-8",
    )
    feature_cache = tmp_path / "features.npz"
    cache_report = tmp_path / "cache-report.json"
    feature_cache.write_bytes(b"not read")
    cache_report.write_bytes(b"not parsed")
    text_model = tmp_path / "text-model"
    text_model.mkdir()
    output = tmp_path / "output"
    config = (
        clevr4_module.Path(__file__).resolve().parents[1]
        / "configs"
        / "pretrained_reuse"
        / "path_a_image_videoprism_clevr4_full.yaml"
    )
    monkeypatch.setattr(
        clevr4_module,
        "_args",
        lambda: SimpleNamespace(
            sample_dir=sample_dir,
            selection_sample_dir=selection_dir,
            feature_cache=feature_cache,
            cache_report=cache_report,
            text_model=text_model,
            config=config,
            output_dir=output,
            cohort_ledger=clevr4_module.DEFAULT_LEDGER,
            seed=17,
            epochs=8,
            batch_questions=64,
        ),
    )
    monkeypatch.setattr(
        clevr4_module, "_features_by_id", lambda *_: pytest.fail("feature cache was loaded")
    )
    monkeypatch.setattr(
        clevr4_module, "_option_embeddings", lambda *_: pytest.fail("text encoder was loaded")
    )

    with pytest.raises(CohortUseBlocked, match="BLOCKED_UNKNOWN_SOURCE_OR_REVISION"):
        clevr4_module.main()

    assert not output.exists()
