import json

import numpy as np
import pytest

from scripts.merge_clevrer_synthetic_av_video_features_v2 import (
    CHECKPOINT_SHA256,
    DATASET_REVISION,
    MODEL_REVISION,
    PREPROCESSING,
    SOURCE_COMMIT,
    merge_features,
    sha256_file,
)


def _cache(path, entries, *, preprocessing=PREPROCESSING):
    path.mkdir()
    hashes = list(entries)
    values = np.stack([entries[key] for key in hashes]).astype(np.float32)
    cache = path / "framewise-features.npz"
    np.savez_compressed(cache, media_sha256=np.asarray(hashes), features=values)
    report = {
        "status": "complete_videoprism_framewise_feature_cache",
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "upstream_source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "frame_count": 16,
        "feature_dimension": 768,
        "feature_layout": "[unique_media, temporal_position, hidden_dimension]",
        "pooling": "mean over spatial token positions separately at each output frame position",
        "preprocessing": preprocessing,
        "feature_cache_sha256": sha256_file(cache),
    }
    (path / "cache-report.json").write_text(json.dumps(report), encoding="utf-8")


def _samples(path, media_hashes):
    path.mkdir()
    rows = [
        {"id": f"row-{i}", "media_sha256": key, "scene_group_id": f"scene-{i}"}
        for i, key in enumerate(media_hashes)
    ]
    (path / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (path / "validation.jsonl").write_text("", encoding="utf-8")
    (path / "fetch-report.json").write_text("{}", encoding="utf-8")


def test_merges_only_requested_media_and_preserves_exact_features(tmp_path):
    first = {"a" * 64: np.full((16, 768), 1, np.float32)}
    second = {
        "b" * 64: np.full((16, 768), 2, np.float32),
        "unused": np.full((16, 768), 3, np.float32),
    }
    _cache(tmp_path / "cache-a", first)
    _cache(tmp_path / "cache-b", second)
    _samples(tmp_path / "samples", ["a" * 64, "b" * 64])

    result = merge_features(
        sample_dir=tmp_path / "samples",
        source_cache_dirs=[tmp_path / "cache-a", tmp_path / "cache-b"],
        output_dir=tmp_path / "out",
    )

    assert result["selected_media_count"] == 2
    with np.load(tmp_path / "out" / "framewise-features.npz", allow_pickle=False) as data:
        assert data["media_sha256"].tolist() == ["a" * 64, "b" * 64]
        assert data["features"][:, 0, 0].tolist() == [1, 2]
    assert result["validation_test_or_sealed_data_opened"] is False


def test_rejects_cache_provenance_mismatch_and_incomplete_coverage(tmp_path):
    _cache(tmp_path / "cache-a", {"a" * 64: np.zeros((16, 768), np.float32)})
    _cache(tmp_path / "cache-b", {"b" * 64: np.zeros((16, 768), np.float32)}, preprocessing="other")
    _samples(tmp_path / "samples", ["a" * 64, "c" * 64])

    with pytest.raises(ValueError, match="provenance"):
        merge_features(
            sample_dir=tmp_path / "samples",
            source_cache_dirs=[tmp_path / "cache-a", tmp_path / "cache-b"],
            output_dir=tmp_path / "out",
        )

    _cache(tmp_path / "cache-b-good", {"b" * 64: np.zeros((16, 768), np.float32)})
    with pytest.raises(ValueError, match="miss 1 selected"):
        merge_features(
            sample_dir=tmp_path / "samples",
            source_cache_dirs=[tmp_path / "cache-a", tmp_path / "cache-b-good"],
            output_dir=tmp_path / "out2",
        )


def test_rejects_overlapping_source_caches_and_existing_output(tmp_path):
    row = np.zeros((16, 768), np.float32)
    _cache(tmp_path / "cache-a", {"a" * 64: row})
    _cache(tmp_path / "cache-b", {"a" * 64: row.copy()})
    _samples(tmp_path / "samples", ["a" * 64])
    with pytest.raises(ValueError, match="overlap"):
        merge_features(
            sample_dir=tmp_path / "samples",
            source_cache_dirs=[tmp_path / "cache-a", tmp_path / "cache-b"],
            output_dir=tmp_path / "out",
        )
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        merge_features(
            sample_dir=tmp_path / "samples",
            source_cache_dirs=[tmp_path / "cache-a", tmp_path / "cache-b"],
            output_dir=tmp_path / "out",
        )
