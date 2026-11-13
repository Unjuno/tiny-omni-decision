from __future__ import annotations

import wave
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
audio_runtime = pytest.importorskip("tiny_omni_decision.audio_runtime")
pair_candidate_features = audio_runtime.pair_candidate_features
read_pcm_wav = audio_runtime.read_pcm_wav


def test_audio_cache_reuses_content_rejects_stale_hash_and_evicts(tmp_path) -> None:
    from tiny_omni_decision.feature_cache import BoundedFeatureCache

    runtime = object.__new__(audio_runtime.SpeechCommandsMultiQueryRuntime)
    runtime.audio_cache = BoundedFeatureCache(1)
    runtime.audio_encoder_calls = 0
    runtime._encode_audio = Mock(side_effect=lambda p: torch.full((384,), p.read_bytes()[0] * 1.0))
    runtime._encode_texts = Mock(return_value={})
    runtime._score_queries = Mock(side_effect=lambda feature, queries, texts: feature.clone())
    queries = [audio_runtime.DecisionQuery("keyword?", ("yes", "no"))]
    path = tmp_path / "live.wav"
    path.write_bytes(b"A")
    original_hash = audio_runtime._file_sha256(path)
    first = runtime.decide_many(path, queries, expected_media_sha256=original_hash)
    alias = tmp_path / "alias.wav"
    alias.write_bytes(b"A")
    torch.testing.assert_close(runtime.decide_many(alias, queries), first)
    assert runtime.audio_cache_metrics["cache_hit"] is True
    assert runtime._encode_audio.call_count == 1
    path.write_bytes(b"B")
    with pytest.raises(ValueError, match="hash differs"):
        runtime.decide_many(path, queries, expected_media_sha256=original_hash)
    assert runtime._encode_audio.call_count == 1
    second = runtime.decide_many(path, queries)
    assert runtime.audio_cache_metrics["cache_hit"] is False
    assert not torch.equal(first, second)
    assert len(runtime.audio_cache) == 1
    runtime.decide_many(alias, queries)
    assert runtime._encode_audio.call_count == 3  # A was evicted by B.


def test_audio_cache_does_not_store_a_file_changed_during_encoding(tmp_path) -> None:
    from tiny_omni_decision.feature_cache import BoundedFeatureCache

    runtime = object.__new__(audio_runtime.SpeechCommandsMultiQueryRuntime)
    runtime.audio_cache = BoundedFeatureCache(1)
    runtime.audio_encoder_calls = 0
    path = tmp_path / "changing.wav"
    path.write_bytes(b"A")

    def encode(source):
        source.write_bytes(b"B")
        return torch.zeros(384)

    runtime._encode_audio = encode
    with pytest.raises(ValueError, match="changed while encoding"):
        runtime.decide_many(path, [audio_runtime.DecisionQuery("q", ("a", "b"))])
    assert len(runtime.audio_cache) == 0


def test_batch_audio_never_reuses_single_event_cache(tmp_path) -> None:
    from tiny_omni_decision.feature_cache import BoundedFeatureCache

    runtime = object.__new__(audio_runtime.SpeechCommandsMultiQueryRuntime)
    runtime.audio_cache = BoundedFeatureCache(1)
    path = tmp_path / "event.wav"
    path.write_bytes(b"same asset")
    runtime.audio_cache.put(audio_runtime._file_sha256(path), torch.ones(384), "single")
    runtime._encode_audio_batch = Mock(return_value=[torch.zeros(384)])
    runtime._encode_texts = Mock(return_value={})
    runtime._score_queries = Mock(side_effect=lambda feature, queries, texts: feature)
    queries = [audio_runtime.DecisionQuery("q", ("a", "b"))]
    output = runtime.decide_batch([path], [queries])
    assert torch.equal(output[0], torch.zeros(384))
    runtime._encode_audio_batch.assert_called_once_with([path])


def test_pair_candidate_features_matches_path_a_layout_and_preserves_option_order() -> None:
    audio = torch.full((384,), 1.0)
    question = torch.full((384,), 2.0)
    options = torch.stack([torch.full((384,), 3.0), torch.full((384,), 4.0)])

    features = pair_candidate_features(audio, question, options)

    assert features.shape == (2, 2688)
    torch.testing.assert_close(features[0, :384], audio)
    torch.testing.assert_close(features[0, 384:768], question)
    torch.testing.assert_close(features[:, 768:1152], options)
    assert torch.all(features[0, 1152:1536] == 1)
    assert torch.all(features[0, 1536:1920] == 2)
    assert torch.all(features[0, 1920:2304] == 6)
    assert torch.all(features[0, 2304:2688] == 3)


def test_pair_candidate_features_rejects_bad_dimensions_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="size 384"):
        pair_candidate_features(torch.zeros(3), torch.zeros(3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="finite"):
        pair_candidate_features(
            torch.zeros(384), torch.zeros(384), torch.full((2, 384), float("nan"))
        )


def test_read_pcm_wav_accepts_the_pinned_mono_16khz_pcm_input(tmp_path) -> None:
    path = tmp_path / "valid.wav"
    samples = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())

    waveform = read_pcm_wav(path)

    assert waveform.shape == (5,)
    assert waveform[0] == pytest.approx(-1.0)
    assert waveform[-1] == pytest.approx(32767 / 32768)


def test_read_pcm_wav_rejects_empty_or_wrong_format(tmp_path) -> None:
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_pcm_wav(empty)

    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.zeros(4, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="mono 16-bit 16 kHz"):
        read_pcm_wav(stereo)
