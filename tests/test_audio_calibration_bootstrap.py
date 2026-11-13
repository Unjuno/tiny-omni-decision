import pytest


def _bootstrap(*args, **kwargs):
    pytest.importorskip("numpy")
    from scripts.bootstrap_textfree_audio_calibration import paired_speaker_bootstrap

    return paired_speaker_bootstrap(*args, **kwargs)


def _rows():
    rows = []
    for speaker, targets in (("speaker-a", [0, 1]), ("speaker-b", [0, 1])):
        for target in targets:
            raw = [0.8, 0.2] if target == 0 else [0.2, 0.8]
            calibrated = [0.95, 0.05] if target == 0 else [0.05, 0.95]
            rows.append(
                {
                    "speaker_group_sha256": speaker,
                    "target_index": target,
                    "probabilities_uncalibrated": raw,
                    "probabilities_temperature_scaled": calibrated,
                }
            )
    return rows


def test_paired_speaker_bootstrap_is_deterministic_and_preserves_argmax():
    first = _bootstrap(_rows(), seed=17, replicates=100)
    second = _bootstrap(_rows(), seed=17, replicates=100)
    assert first == second
    assert first["paired_differences"]["accuracy"]["percentile_95_ci"] == [0.0, 0.0]
    assert first["paired_differences"]["nll"]["percentile_95_ci"][1] < 0.0


def test_paired_speaker_bootstrap_rejects_nonprobability_rows():
    rows = _rows()
    rows[0]["probabilities_uncalibrated"] = [0.5, 0.6]
    with pytest.raises(ValueError, match="probability"):
        _bootstrap(rows, seed=17, replicates=10)
