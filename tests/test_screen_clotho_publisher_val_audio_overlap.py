import numpy as np

from scripts.audio_overlap_probe import PreparedWaveformMatcher
from scripts.screen_clotho_publisher_val_audio_overlap import oriented_match, wave_sha256


def test_waveform_hash_uses_canonical_little_endian_float32():
    values = np.asarray([0.125, -0.25, 0.5], dtype=np.float32)

    assert wave_sha256(values) == wave_sha256(values.astype(np.float64))


def test_oriented_match_uses_shorter_clip_as_template():
    template = np.sin(np.arange(32, dtype=np.float32) / 3)
    recording = np.concatenate(
        (np.zeros(10, dtype=np.float32), template, np.ones(12, dtype=np.float32))
    )
    matcher = PreparedWaveformMatcher({"short": template, "long": recording}, 1_000_000)

    result = oriented_match(matcher, "long", recording, "short", template)

    assert result["absolute_correlation"] > 0.999
    assert result["offset_samples"] == 10
