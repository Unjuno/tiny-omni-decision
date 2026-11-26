from __future__ import annotations

import pytest

from tiny_omni_decision.feature_cache import BoundedFeatureCache


def test_feature_cache_returns_cached_features_and_refreshes_lru_order() -> None:
    cache = BoundedFeatureCache(max_entries=2)
    cache.put("image:a", [1.0], "sha-a")
    cache.put("video:b", [2.0], "sha-b")

    assert cache.get("image:a") == ([1.0], "sha-a")
    cache.put("image:c", [3.0], "sha-c")

    assert len(cache) == 2
    assert cache.get("video:b") is None
    assert cache.get("image:a") == ([1.0], "sha-a")
    assert cache.get("image:c") == ([3.0], "sha-c")


def test_feature_cache_requires_positive_capacity_and_content_identity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        BoundedFeatureCache(max_entries=0)

    cache = BoundedFeatureCache(max_entries=1)
    with pytest.raises(ValueError, match="must be nonempty"):
        cache.put("", [1.0], "sha-a")
    with pytest.raises(ValueError, match="must be nonempty"):
        cache.put("image:a", [1.0], "")
