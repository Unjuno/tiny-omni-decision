from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_frozen_videoprism_clevr4_probe import (  # noqa: E402
    _features_by_id,
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
