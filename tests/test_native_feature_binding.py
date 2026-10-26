import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tiny_omni_decision.cache import ObservationFeatureCache, ObservationFeatureKey
from tiny_omni_decision.native_feature_binding import NativeFeatureBinding, load_bound_feature
from tiny_omni_decision.observations import Observation


def fixture(tmp_path, modality="image", shape=(2,)):
    media = tmp_path / "input.bin"
    media.write_bytes(b"original observation")
    media_sha = hashlib.sha256(media.read_bytes()).hexdigest()
    values = np.arange(np.prod(shape), dtype="<f4").reshape(shape)
    key = ObservationFeatureKey(
        modality=modality,
        source_id="dataset",
        source_revision="1" * 40,
        observation_sha256=media_sha,
        encoder_id="encoder",
        encoder_revision="2" * 40,
        encoder_weights_sha256="3" * 64,
        preprocessor_id="processor",
        preprocessor_revision="2" * 40,
        preprocessing_sha256="4" * 64,
        feature_name="frozen_features",
        feature_dtype="float32",
        feature_shape=shape,
        temporal_policy="eight ordered frames" if modality == "video" else None,
    )
    cache = ObservationFeatureCache(tmp_path / "cache")
    entry = cache.put(key, values.tobytes())
    times = tuple(float(i) for i in range(shape[0])) if modality == "video" else (0.0,)
    binding = NativeFeatureBinding(
        key=key,
        feature_payload_sha256=entry.payload_sha256,
        source_id="input-1",
        clock_id="local",
        timestamps_ms=times,
    )
    obs = Observation(
        modality=modality,
        source_id="input-1",
        payload_ref=str(media),
        payload_sha256=media_sha,
        clock_id="local",
        timestamps_ms=times,
        valid_mask=(True,) * len(times),
    )
    return media, values, cache, binding, obs


def test_bound_feature_reads_verified_payload_and_allows_same_content_alias(tmp_path):
    media, values, cache, binding, obs = fixture(tmp_path)
    actual = load_bound_feature(obs, binding, cache)
    np.testing.assert_array_equal(actual, values)
    actual[0] = 99
    np.testing.assert_array_equal(load_bound_feature(obs, binding, cache), values)
    alias = tmp_path / "alias.bin"
    alias.write_bytes(media.read_bytes())
    np.testing.assert_array_equal(
        load_bound_feature(obs.model_copy(update={"payload_ref": str(alias)}), binding, cache),
        values,
    )


def test_same_path_same_length_replacement_cannot_reuse_old_features(tmp_path):
    media, _, cache, binding, obs = fixture(tmp_path)
    import os

    stat = media.stat()
    media.write_bytes(b"X" * stat.st_size)
    os.utime(media, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="media.*SHA"):
        load_bound_feature(obs, binding, cache)


@pytest.mark.parametrize(
    "field,value",
    [
        ("modality", "audio"),
        ("source_id", "other"),
        ("clock_id", "other-clock"),
        ("timestamps_ms", (1.0,)),
        ("payload_sha256", "f" * 64),
        ("payload_index", 0),
    ],
)
def test_declared_observation_cannot_change_trusted_binding(tmp_path, field, value):
    _, _, cache, binding, obs = fixture(tmp_path)
    with pytest.raises(ValueError):
        load_bound_feature(obs.model_copy(update={field: value}), binding, cache)


def test_valid_cache_entry_with_different_feature_bytes_is_rejected(tmp_path):
    _, values, _, binding, obs = fixture(tmp_path)
    forged = ObservationFeatureCache(tmp_path / "other-cache")
    forged.put(binding.key, (values + 1).tobytes())
    with pytest.raises(ValueError, match="feature.*SHA"):
        load_bound_feature(obs, binding, forged)


def test_changed_encoder_or_preprocessing_cannot_fall_back_to_old_cache(tmp_path):
    _, _, cache, binding, obs = fixture(tmp_path)
    for field in ("encoder_weights_sha256", "preprocessing_sha256"):
        different = replace(binding, key=replace(binding.key, **{field: "e" * 64}))
        with pytest.raises(FileNotFoundError):
            load_bound_feature(obs, different, cache)


def test_nan_or_wrong_shape_payload_fails_even_with_matching_feature_hash(tmp_path):
    _, _, cache, binding, obs = fixture(tmp_path)
    for index, values in enumerate(
        (np.array([np.nan, 0], dtype="<f4"), np.array([0], dtype="<f4"))
    ):
        other = ObservationFeatureCache(tmp_path / f"bad-cache-{index}")
        entry = other.put(binding.key, values.tobytes())
        with pytest.raises(ValueError):
            load_bound_feature(
                obs, replace(binding, feature_payload_sha256=entry.payload_sha256), other
            )


def test_bound_video_checks_time_count_against_feature_sequence(tmp_path):
    _, values, cache, binding, obs = fixture(tmp_path, "video", (8, 2))
    np.testing.assert_array_equal(load_bound_feature(obs, binding, cache), values)
    with pytest.raises(ValueError, match="timestamp"):
        replace(binding, timestamps_ms=(0.0,))


@pytest.mark.parametrize("modality", ["audio", "image", "video"])
def test_real_readout_bound_json_matches_direct_features(tmp_path, modality):
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.native_feature_binding import decide_bound_native
    from tiny_omni_decision.observations import ObservationBatch
    from tiny_omni_decision.textfree_audio import TextFreeAudioReadout
    from tiny_omni_decision.textfree_image import TextFreeImageReadout
    from tiny_omni_decision.textfree_video import TextFreeVideoReadout, build_video_vocab

    torch.manual_seed(17)
    shape = (8, 768) if modality == "video" else (2,)
    _, values, cache, binding, obs = fixture(tmp_path, modality, shape)
    batch = ObservationBatch(
        schema_version="1.0",
        observations=(obs,),
        options=(
            {"option_id": "first", "attributes": {"class_id": "0"}},
            {"option_id": "second", "attributes": {"class_id": "1"}},
        ),
    )
    if modality == "audio":
        model = TextFreeAudioReadout(2, ["0", "1"]).eval()
        expected = model.decide_native(torch.from_numpy(values), batch)
        kwargs = {}
    elif modality == "image":
        model = TextFreeImageReadout(2, {"count": ["0", "1"]}).eval()
        expected = model.decide_native(torch.from_numpy(values), "count", batch)
        kwargs = {"taxonomy": "count"}
    else:
        tokens = ["events", "count"]
        vocab = build_video_vocab([{"program": tokens, "taxonomy": "count", "options": ["0", "1"]}])
        model = TextFreeVideoReadout(vocab, temporal_mode="bilstm").eval()
        kwargs = {"taxonomy": "count", "program_tokens": tokens}
        expected = model.decide_native(torch.from_numpy(values), batch, **kwargs)
    actual = decide_bound_native(model, batch, binding, cache, **kwargs)
    provenance = actual.pop("feature_provenance")
    assert actual == expected
    assert provenance["cache_id"] == binding.key.cache_id
    assert provenance["feature_payload_sha256"] == binding.feature_payload_sha256


def test_incomplete_request_abstains_without_reading_missing_media_or_cache(tmp_path):
    pytest.importorskip("torch")
    from tiny_omni_decision.native_feature_binding import decide_bound_native
    from tiny_omni_decision.observations import ObservationBatch
    from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

    media, _, _, binding, obs = fixture(tmp_path, "audio")
    media.unlink()
    batch = ObservationBatch(
        schema_version="1.0",
        observations=(obs.model_copy(update={"valid_mask": (False,)}),),
        options=(
            {"option_id": "a", "attributes": {"class_id": "0"}},
            {"option_id": "b", "attributes": {"class_id": "1"}},
        ),
    )
    model = TextFreeAudioReadout(2, ["0", "1"]).eval()
    result = decide_bound_native(
        model, batch, binding, ObservationFeatureCache(tmp_path / "missing")
    )
    assert result["status"] == "abstain"
    assert "feature_provenance" not in result
