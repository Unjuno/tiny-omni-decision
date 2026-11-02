from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="the optional CPU probe runtime requires PyTorch")
probe = importlib.import_module("scripts.train_inaturalist_videoprism_linear_probe")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, Path]:
    feature_dir = root / "feature-cache"
    metadata_dir = root / "metadata"
    feature_dir.mkdir()
    metadata_dir.mkdir()
    train_rows = []
    validation_rows = []
    features = []
    feature_rows = []
    for split, repetitions in (("train", 2), ("validation", 1)):
        split_rows = []
        for class_id in range(8):
            for repeat in range(repetitions):
                observation_id = f"{split}-{class_id}-{repeat}"
                row = {"taxon_id": class_id + 100, "observation_id": observation_id}
                split_rows.append(row)
                vector = np.zeros(768, dtype=np.float32)
                vector[class_id] = 5.0
                features.append(vector)
                feature_rows.append(
                    {
                        "split": split,
                        "sample_id": f"inat:{class_id + 100}:{observation_id}",
                        "taxon_id": class_id + 100,
                        "observation_id": observation_id,
                        "photo_id": f"photo-{observation_id}",
                    }
                )
        if split == "train":
            train_rows = split_rows
        else:
            validation_rows = split_rows
        (metadata_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in split_rows), encoding="utf-8"
        )
    cache_path = feature_dir / "features.npy"
    np.save(cache_path, np.stack(features))
    rows_path = feature_dir / "feature-rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8")
    result = {
        "status": "complete",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "feature_cache_sha256": _sha(cache_path),
        "feature_rows_sha256": _sha(rows_path),
        "metadata_sha256": {
            split: _sha(metadata_dir / f"{split}.jsonl") for split in ("train", "validation")
        },
    }
    result_path = feature_dir / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (feature_dir / "run-status.json").write_text(
        json.dumps({"status": "COMPLETE", "result_sha256": _sha(result_path)}),
        encoding="utf-8",
    )
    return feature_dir, metadata_dir


def test_load_verified_data_preserves_split_order_and_class_ids(tmp_path: Path) -> None:
    feature_dir, metadata_dir = _write_fixture(tmp_path)

    x_train, y_train, x_val, y_val, class_ids = probe.load_verified_data(
        feature_dir, metadata_dir
    )

    assert x_train.shape == (16, 768)
    assert x_val.shape == (8, 768)
    assert y_train.tolist() == [index for index in range(8) for _ in range(2)]
    assert y_val.tolist() == list(range(8))
    assert class_ids == list(range(100, 108))


def test_load_verified_data_rejects_changed_feature_row_order(tmp_path: Path) -> None:
    feature_dir, metadata_dir = _write_fixture(tmp_path)
    path = feature_dir / "feature-rows.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="feature row hash mismatch"):
        probe.load_verified_data(feature_dir, metadata_dir)


def test_cpu_probe_writes_selected_checkpoint_and_complete_predictions(tmp_path: Path) -> None:
    feature_dir, metadata_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "readout"

    result = probe.run(feature_dir, metadata_dir, output_dir, epochs=2, patience=2)

    assert result["status"] == "complete_inaturalist_videoprism_linear_probe"
    assert result["device"] == "cpu"
    assert result["trainable_parameters"] == 8 * 768 + 8
    assert result["train_count"] == 16
    assert result["validation_count"] == 8
    assert (output_dir / "best-linear-readout.pt").is_file()
    predictions = (output_dir / "validation-predictions.jsonl").read_text(encoding="utf-8")
    assert len(predictions.splitlines()) == 8
