from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.pack_holoassist_feature_cache import (
    load_feature_bank,
    pack_split,
    select_split_banks,
)


def metadata(name: str) -> dict[str, str]:
    return {
        "model_id": name,
        "revision": "fixed-revision",
        "weights_sha256": "0" * 64,
        "loaded_weights_sha256": "0" * 64,
        "preprocessing": "deterministic test fixture",
    }


def make_rows():
    return [
        {
            "id": "train-a:window-0000",
            "split": "train",
            "session_id": "train-a",
            "window_index": 0,
            "target": 0,
            "media_sha256": "a" * 64,
            "task_type": "assemble",
        },
        {
            "id": "train-a:window-0001",
            "split": "train",
            "session_id": "train-a",
            "window_index": 1,
            "target": None,
            "media_sha256": "a" * 64,
            "task_type": "assemble",
        },
    ]


def test_pack_split_aligns_frozen_banks_and_masks_unlabelled_context() -> None:
    rows = make_rows()
    ids = [row["id"] for row in rows]
    banks = {
        "video": (ids, np.ones((2, 768), dtype=np.float32), metadata("video")),
        "audio": (ids, np.full((2, 384), 2.0, dtype=np.float32), metadata("audio")),
        "text": (ids, np.full((2, 384), 3.0, dtype=np.float32), metadata("text")),
    }

    packed, report = pack_split(rows, banks, "train")

    assert packed["video"].shape == (2, 768)
    assert packed["audio"][0, 0] == 2.0
    assert packed["text"][1, 0] == 3.0
    assert packed["target"].tolist() == [0, -1]
    assert packed["loss_mask"].tolist() == [True, False]
    assert report["session_count"] == 1
    assert report["unlabelled_context_window_count"] == 1


def test_pack_split_rejects_encoder_order_mismatch() -> None:
    rows = make_rows()
    ids = [row["id"] for row in rows]
    banks = {
        "video": (ids[::-1], np.ones((2, 768), dtype=np.float32), metadata("video")),
        "audio": (ids, np.ones((2, 384), dtype=np.float32), metadata("audio")),
        "text": (ids, np.ones((2, 384), dtype=np.float32), metadata("text")),
    }

    with pytest.raises(ValueError, match="ID order"):
        pack_split(rows, banks, "train")


def test_select_split_banks_slices_global_bank_in_window_order() -> None:
    train_rows = make_rows()
    validation_id = "validation-a:window-0000"
    global_ids = [row["id"] for row in train_rows] + [validation_id]
    global_features = np.arange(3 * 768, dtype=np.float32).reshape(3, 768)
    banks = {
        "video": (global_ids, global_features, metadata("video")),
    }

    selected = select_split_banks(train_rows, banks)

    assert selected["video"][0] == [row["id"] for row in train_rows]
    np.testing.assert_array_equal(selected["video"][1], global_features[:2])


def test_feature_bank_loader_requires_finite_pinned_embedding(tmp_path) -> None:
    path = tmp_path / "video.npz"
    np.savez_compressed(
        path,
        sample_id=np.asarray(["id-1"], dtype=str),
        features=np.full((1, 768), np.nan, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata("video"))),
    )

    with pytest.raises(ValueError, match="non-finite"):
        load_feature_bank(path, "video")
