from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from scripts.verify_video_lru_scorer_parity import _fetch_feature
from tiny_omni_decision.cache import (
    InMemoryObservationFeatureLRU,
    ObservationFeatureCache,
    ObservationFeatureKey,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def _key(observation: bytes) -> ObservationFeatureKey:
    digest = hashlib.sha256(b"pinned").hexdigest()
    return ObservationFeatureKey(
        modality="video",
        source_id="MIT-IBM/CLEVRER",
        source_revision="98b842082ba4f7c18b6b9e3f39145871782a65ef",
        observation_sha256=hashlib.sha256(observation).hexdigest(),
        encoder_id="facebookresearch/vjepa2/vjepa2_1_vit_base_384",
        encoder_revision="204698b45b3712590f06245fbfba32d3be539812",
        encoder_weights_sha256=digest,
        preprocessor_id="vjepa2_preprocessor",
        preprocessor_revision="204698b45b3712590f06245fbfba32d3be539812",
        preprocessing_sha256=digest,
        feature_name="mean_spatiotemporal_tokens",
        feature_dtype="float32",
        feature_shape=(4,),
        temporal_policy="uniform-linspace-8-inclusive-v1",
    )


def test_lru_features_feed_candidate_scorer_with_exact_direct_parity(tmp_path):
    torch.manual_seed(17)
    persistent = ObservationFeatureCache(tmp_path)
    keys = [_key(b"scene-a"), _key(b"scene-b"), _key(b"scene-c")]
    features = [
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.tensor([0.5, 0.6, 0.7, 0.8]),
        torch.tensor([0.9, 1.0, 1.1, 1.2]),
    ]
    entries = [
        persistent.put(key, feature.numpy().tobytes())
        for key, feature in zip(keys, features, strict=True)
    ]
    memory = InMemoryObservationFeatureLRU(max_bytes=32)
    scorer = FrozenFeatureCandidateScorer(feature_size=6, hidden_size=8).eval()
    question_and_option_features = torch.tensor(
        [[0.25, 0.75], [0.9, 0.1]], dtype=torch.float32
    )

    observed = []
    for key, entry in zip(
        [keys[0], keys[0], keys[1], keys[2]],
        [entries[0], entries[0], entries[1], entries[2]],
        strict=True,
    ):
        feature, _ = _fetch_feature(key, entry.payload_sha256, memory, persistent)
        observed.append(feature)

    assert memory.hit_count == 1
    assert memory.miss_count == 3
    assert memory.eviction_count == 1

    direct = torch.cat([features[0].expand(2, -1), question_and_option_features], dim=-1)
    cached = torch.cat([observed[1].expand(2, -1), question_and_option_features], dim=-1)
    direct_output = scorer.decide(direct)
    cached_output = scorer.decide(cached)

    assert torch.equal(direct_output.option_logits, cached_output.option_logits)
    assert torch.equal(direct_output.probabilities, cached_output.probabilities)
    assert direct_output.predicted_index == cached_output.predicted_index
