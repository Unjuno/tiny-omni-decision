from collections import Counter, defaultdict
from pathlib import Path

import pytest

from tiny_omni_decision.synthetic_av_task import build_manifest, validate_manifest


def _render_assets_with_optional_video_dependency():
    pytest.importorskip("av")
    from scripts.prepare_synthetic_av_readout import render_assets

    return render_assets


def test_manifest_is_deterministic_and_query_polarities_are_complements():
    manifest = build_manifest(train_groups=4, validation_groups=2, seed=17)
    validate_manifest(manifest)
    assert manifest == build_manifest(train_groups=4, validation_groups=2, seed=17)
    rows = manifest["examples"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    assert all(len(pair) == 2 for pair in grouped.values())
    assert all({row["target"] for row in pair} == {0, 1} for pair in grouped.values())


def test_manifest_makes_hard_negatives_with_same_multisets_and_balanced_marginals():
    manifest = build_manifest(train_groups=12, validation_groups=4, seed=17)
    rows = manifest["examples"]
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
        assert Counter(row["audio_symbols"]) == Counter(row["video_symbols"])
    for group_rows in by_group.values():
        assert len(group_rows) == 8
        for polarity in {row["query_polarity"] for row in group_rows}:
            selected = [row for row in group_rows if row["query_polarity"] == polarity]
            assert Counter(row["target"] for row in selected) == Counter({0: 2, 1: 2})


def test_manifest_rejects_sequence_identity_crossing_splits():
    manifest = build_manifest(train_groups=2, validation_groups=2, seed=17)
    validation_group = next(group for group in manifest["groups"] if group["split"] == "validation")
    train_group = next(group for group in manifest["groups"] if group["split"] == "train")
    validation_group["sequence_ids"]["A"] = train_group["sequence_ids"]["A"]
    with pytest.raises(ValueError, match="sequence crosses splits"):
        validate_manifest(manifest)


def test_manifest_rejects_missing_pair_examples():
    manifest = build_manifest(train_groups=2, validation_groups=2, seed=17)
    manifest["examples"].pop()
    with pytest.raises(ValueError, match="every group must have exactly eight examples"):
        validate_manifest(manifest)


def test_renderer_is_deterministic_and_persists_verified_av_assets(tmp_path: Path):
    render_assets = _render_assets_with_optional_video_dependency()
    manifest = build_manifest(train_groups=1, validation_groups=1, seed=17)
    first = render_assets(manifest, tmp_path / "first")
    second = render_assets(manifest, tmp_path / "second")
    first_assets = first["assets"]
    second_assets = second["assets"]
    assert first_assets == second_assets
    assert len(first_assets["videos"]) == 4
    assert len(first_assets["audio"]) == 4
    assert len(first["examples"]) == 16
    assert all(item["frame_count"] == 8 for item in first_assets["videos"].values())
    assert all(item["sample_rate_hz"] == 32_000 for item in first_assets["audio"].values())
    audio_paths = (tmp_path / "first" / item["path"] for item in first_assets["audio"].values())
    assert all(path.is_file() for path in audio_paths)


def test_video_extraction_decoder_preserves_every_rendered_symbol_in_order(tmp_path: Path):
    render_assets = _render_assets_with_optional_video_dependency()
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from scripts.extract_frozen_videoprism_features import _decode_video

    manifest = build_manifest(train_groups=1, validation_groups=1, seed=17)
    rendered = render_assets(manifest, tmp_path / "ordered")
    for sequence_id, record in rendered["assets"]["videos"].items():
        decoded = _decode_video(tmp_path / "ordered" / record["path"])
        expected = np.stack(
            [
                np.asarray(Image.open(tmp_path / "ordered" / frame_path).convert("RGB"))
                for frame_path in record["frame_paths"]
            ]
        )

        assert decoded.shape == (len(record["symbols"]), record["height"], record["width"], 3)
        assert np.array_equal(
            decoded, expected
        ), f"decoded video frame order differs: {sequence_id}"


def test_rendered_audio_symbols_keep_order_and_align_with_video_frames(tmp_path: Path):
    render_assets = _render_assets_with_optional_video_dependency()
    np = pytest.importorskip("numpy")
    import wave

    from scripts.prepare_synthetic_av_readout import (
        EVENT_SECONDS,
        FRAME_RATE,
        SAMPLE_RATE_HZ,
        TONE_FREQUENCIES_HZ,
    )

    manifest = build_manifest(train_groups=1, validation_groups=1, seed=17)
    rendered = render_assets(manifest, tmp_path / "aligned")
    for sequence_id, audio_record in rendered["assets"]["audio"].items():
        video_record = rendered["assets"]["videos"][sequence_id]
        segments = audio_record["segments"]
        assert video_record["frame_count"] == len(video_record["symbols"])
        assert FRAME_RATE * EVENT_SECONDS == 1
        assert audio_record["duration_seconds"] == video_record["frame_count"] / FRAME_RATE
        assert len(segments) == len(video_record["symbols"])

        with wave.open(str(tmp_path / "aligned" / audio_record["path"]), "rb") as source:
            assert source.getframerate() == SAMPLE_RATE_HZ
            assert source.getnchannels() == 1
            waveform = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")

        decoded_symbols = []
        for index, segment in enumerate(segments):
            start = index * round(SAMPLE_RATE_HZ * EVENT_SECONDS)
            stop = start + round(SAMPLE_RATE_HZ * EVENT_SECONDS)
            samples = waveform[start:stop].astype(np.float64)
            assert segment["sample_start"] == start
            assert segment["sample_end"] == stop
            assert len(samples) == round(SAMPLE_RATE_HZ * EVENT_SECONDS)
            spectrum = np.abs(np.fft.rfft((samples - samples.mean()) * np.hanning(len(samples))))
            frequencies = np.fft.rfftfreq(len(samples), d=1 / SAMPLE_RATE_HZ)
            peak_hz = frequencies[1 + np.argmax(spectrum[1:])]
            symbol = int(np.argmin(np.abs(np.asarray(TONE_FREQUENCIES_HZ) - peak_hz)))
            decoded_symbols.append(symbol)

            assert segment["event_index"] == index
            assert segment["symbol_id"] == video_record["symbols"][index]
            assert segment["start_seconds"] == index / FRAME_RATE
            assert segment["end_seconds"] == (index + 1) / FRAME_RATE
            assert abs(peak_hz - TONE_FREQUENCIES_HZ[symbol]) <= 2.0

        assert decoded_symbols == video_record["symbols"], sequence_id


def test_renderer_refuses_to_overwrite_an_existing_output(tmp_path: Path):
    render_assets = _render_assets_with_optional_video_dependency()
    output = tmp_path / "occupied"
    output.mkdir()
    manifest = build_manifest(train_groups=1, validation_groups=1, seed=17)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_assets(manifest, output)
