from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.finalize_physicalai_feature_metadata import finalize_metadata


def test_finalizer_binds_metadata_to_validation_receipt_and_features(tmp_path: Path):
    (tmp_path / "download-receipt.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "manifest_sha256": "b" * 64,
                "validation_episode_indices": [12, 17],
            }
        )
    )
    np.savez_compressed(
        tmp_path / "features.npz",
        episode=np.array([12] * 4 + [17] * 4),
        split=np.array(["validation"] * 8),
        t=np.arange(8),
    )

    metadata = finalize_metadata(
        tmp_path,
        device="cuda",
        batch_size=1,
        decode_seconds=2.0,
        inference_seconds=3.0,
        peak_cuda_bytes=None,
    )

    assert metadata["validation_episode_count"] == 2
    assert metadata["validation_query_count"] == 8
    assert metadata["reserved_holdout_accessed"] is False
    assert len(metadata["feature_file_sha256"]) == 64


def test_finalizer_rejects_wrong_episode_membership(tmp_path: Path):
    (tmp_path / "download-receipt.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "manifest_sha256": "b" * 64,
                "validation_episode_indices": [12],
            }
        )
    )
    np.savez_compressed(
        tmp_path / "features.npz",
        episode=np.array([13] * 4),
        split=np.array(["validation"] * 4),
        t=np.arange(4),
    )

    with pytest.raises(ValueError, match="episode IDs"):
        finalize_metadata(
            tmp_path,
            device="cpu",
            batch_size=1,
            decode_seconds=None,
            inference_seconds=None,
            peak_cuda_bytes=None,
        )
