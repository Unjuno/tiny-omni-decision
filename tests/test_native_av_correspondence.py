import hashlib

import numpy as np
import pytest


def runtime():
    from tiny_omni_decision.native_av_correspondence import NativeAVCorrespondence

    calls = []

    def backend(kind):
        def encode(data):
            calls.append((kind, data))
            return np.random.default_rng(data[0]).normal(size=51).astype(np.float32)

        return encode

    model = NativeAVCorrespondence(
        backend("audio"), backend("video"), np.ones(6), np.ones(6), 1.0, "a" * 64, max_entries=8
    )
    return model, calls


def test_raw_content_cache_reuse_and_exact_reference_json():
    from scripts.lombard_correspondence import agreement_scores, choice_json

    model, calls = runtime()
    audio = model.encode("audio", b"query", time_range=(0.0, 1.0))
    videos = [model.encode("video", bytes([i]), time_range=(0.0, 1.0)) for i in range(4)]
    assert model.encode("audio", b"query", time_range=(0.0, 1.0)) is audio
    ids = ["z", "w", "x", "y"]
    actual = model.decide(audio, videos, ids)
    ref = agreement_scores(
        np.array([audio.logits]), np.array([[v.logits for v in videos]]), np.ones(6), np.ones(6)
    )[0]
    assert actual == choice_json(ref, ids, 1)
    assert actual == model.decide(audio, videos[::-1], ids[::-1])
    assert len(calls) == 5 and model.cache_hits == 1
    assert audio.media_sha256 == hashlib.sha256(b"query").hexdigest()


def test_owner_profile_time_and_media_key_fail_closed():
    model, calls = runtime()
    a = model.encode("audio", b"query", time_range=(0.0, 1.0))
    b = model.encode("audio", b"query", time_range=(1.0, 2.0))
    assert a is not b and len(calls) == 2
    v = [model.encode("video", bytes([i]), time_range=(0.0, 1.0)) for i in range(4)]
    with pytest.raises(ValueError):
        model.encode("audio", b"query", time_range=(0, 1), expected_sha256="0" * 64)
    with pytest.raises(ValueError):
        runtime()[0].decide(a, v, ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        model.decide(a, v, ["same"] * 4)
    with pytest.raises(ValueError):
        model.encode("text", b"a", time_range=(0, 1))
    with pytest.raises(ValueError):
        model.encode("audio", b"a", time_range=(1, 0))
    with pytest.raises(ValueError):
        model.encode("audio", b"", time_range=(0, 1))


def test_bounded_lru_and_explicit_missing_policy():
    import json

    model, calls = runtime()
    for i in range(9):
        model.encode("audio", bytes([i]), time_range=(0, 1))
    assert model.cache_entries == 8
    model.encode("audio", b"\x00", time_range=(0, 1))
    assert len(calls) == 10
    assert (
        list(json.loads(model.decide(None, None, ["a", "b", "c", "d"]))["probabilities"].values())
        == [0.25] * 4
    )
