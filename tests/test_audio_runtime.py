from __future__ import annotations

import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")
audio_runtime = pytest.importorskip("tiny_omni_decision.audio_runtime")
pair_candidate_features = audio_runtime.pair_candidate_features
read_pcm_wav = audio_runtime.read_pcm_wav


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
