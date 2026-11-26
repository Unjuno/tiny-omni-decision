import hashlib

import numpy as np
import pytest

from scripts.audit_native_mask_memory_smoke import verify_raw_archive


def archive():
    raw = np.full((2, 32, 1, 1, 256, 256), -1, np.float32)
    scores = np.zeros((2, 32, 1, 1), np.float32)
    digests = []
    for i in range(2):
        h = hashlib.sha256()
        for t in range(32):
            h.update(raw[i, t].tobytes())
            h.update(scores[i, t].tobytes())
        digests.append(h.hexdigest())
    return dict(
        raw_masks=raw,
        raw_scores=scores,
        masks_packed=np.zeros((2, 32, 8192), np.uint8),
        boxes=np.zeros((2, 32, 4), np.float64),
        empty_masks=np.ones((2, 32), bool),
        scores=scores.reshape(2, 32),
        raw_sha256=np.array(digests),
        seed_indices=np.array([9, 2], np.int64),
        frame_indices=np.arange(32, dtype=np.int64),
    )


def test_independent_raw_pack_geometry_stream_and_empty_accounting():
    a = archive()
    assert verify_raw_archive(a, [9, 2], list(range(32))) == dict(seeds=2, frames=64, empty=64)
    a["boxes"][0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        verify_raw_archive(a, [9, 2], list(range(32)))


def test_wrong_raw_digest_and_identity_are_rejected():
    a = archive()
    a["raw_sha256"][0] = "a" * 64
    with pytest.raises(ValueError):
        verify_raw_archive(a, [9, 2], list(range(32)))
    with pytest.raises(ValueError):
        verify_raw_archive(archive(), [2, 9], list(range(32)))
