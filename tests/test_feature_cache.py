from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tiny_omni_decision.cache import (
    FeatureCacheError,
    ObservationFeatureCache,
    ObservationFeatureKey,
)
from tiny_omni_decision.packed_cache import PackedObservationFeatureCache


def _key() -> ObservationFeatureKey:
    digest = hashlib.sha256(b"pinned").hexdigest()
    return ObservationFeatureKey(
        modality="video",
        source_id="MIT-IBM/CLEVRER",
        source_revision="98b842082ba4f7c18b6b9e3f39145871782a65ef",
        observation_sha256=hashlib.sha256(b"scene-video").hexdigest(),
        encoder_id="facebookresearch/vjepa2/vjepa2_1_vit_base_384",
        encoder_revision="204698b45b3712590f06245fbfba32d3be539812",
        encoder_weights_sha256=digest,
        preprocessor_id="vjepa2_preprocessor",
        preprocessor_revision="204698b45b3712590f06245fbfba32d3be539812",
        preprocessing_sha256=digest,
        feature_name="mean_spatiotemporal_tokens",
        feature_dtype="float32",
        feature_shape=(768,),
        temporal_policy="uniform-linspace-8-inclusive-v1",
    )


def test_observation_feature_cache_roundtrip_is_immutable_and_content_checked(tmp_path):
    cache = ObservationFeatureCache(tmp_path)
    key = _key()
    payload = bytes(range(256))

    written = cache.put(key, payload)
    repeated = cache.put(key, payload)
    loaded = cache.get(key)

    assert written.key.cache_id == key.cache_id
    assert repeated.payload == loaded.payload == payload
    assert loaded.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert loaded.entry_bytes > len(payload)
    with pytest.raises(FeatureCacheError, match="different feature payload"):
        cache.put(key, b"different")


@pytest.mark.parametrize(
    "changed",
    [
        {"observation_sha256": hashlib.sha256(b"other-scene").hexdigest()},
        {"source_revision": "other-source-revision"},
        {"encoder_weights_sha256": hashlib.sha256(b"other-model").hexdigest()},
        {"preprocessing_sha256": hashlib.sha256(b"other-processor").hexdigest()},
        {"temporal_policy": "uniform-linspace-12-inclusive-v1"},
    ],
)
def test_observation_cache_identity_invalidates_on_input_or_pipeline_change(changed):
    original = _key()
    updated = replace(original, **changed)

    assert original.cache_id != updated.cache_id


def test_observation_feature_cache_rejects_payload_corruption(tmp_path):
    cache = ObservationFeatureCache(tmp_path)
    key = _key()
    cache.put(key, b"stable feature bytes")
    path = tmp_path / f"{key.cache_id}.feature"
    path.write_bytes(path.read_bytes()[:-1] + b"!")

    with pytest.raises(FeatureCacheError, match="SHA-256 mismatch"):
        cache.get(key)


@pytest.mark.parametrize(
    "updates",
    [
        {"modality": "unknown"},
        {"encoder_weights_sha256": "not-a-hash"},
        {"feature_shape": (768, 0)},
        {"start_ms": 900, "end_ms": 100},
        {"temporal_policy": None},
    ],
)
def test_observation_feature_key_rejects_invalid_identity(updates):
    with pytest.raises(FeatureCacheError):
        replace(_key(), **updates)


def test_packed_observation_cache_preserves_entries_and_is_immutable(tmp_path):
    source = tmp_path / "source"
    source_cache = ObservationFeatureCache(source)
    keys = [_key(), replace(_key(), observation_sha256=hashlib.sha256(b"other").hexdigest())]
    payloads = [b"first feature", b"second feature"]
    expected = [
        source_cache.put(key, payload)
        for key, payload in zip(keys, payloads, strict=True)
    ]
    target = tmp_path / "packed"

    with PackedObservationFeatureCache.pack(source, target) as packed:
        packed.verify()
        assert packed.manifest["entry_count"] == 2
        assert [packed.get(key) for key in keys] == expected
        with pytest.raises(FileNotFoundError, match="not found"):
            packed.get(replace(_key(), observation_sha256=hashlib.sha256(b"missing").hexdigest()))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        PackedObservationFeatureCache.pack(source, target)
    assert len(list(source.glob("*.feature"))) == 2


def test_packed_observation_cache_detects_entry_and_container_corruption(tmp_path):
    source = tmp_path / "source"
    key = _key()
    ObservationFeatureCache(source).put(key, b"stable feature")
    target = tmp_path / "packed"
    packed = PackedObservationFeatureCache.pack(source, target)
    pack_path = target / "features.pack"
    pack_path.write_bytes(pack_path.read_bytes()[:-1] + b"!")

    with pytest.raises(FeatureCacheError, match="entry is truncated or has changed"):
        packed.get(key)
    with pytest.raises(FeatureCacheError, match="SHA-256 does not match manifest"):
        packed.verify()
    packed.close()
