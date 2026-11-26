from __future__ import annotations

import json
import wave
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from scripts.extract_perception_temporal_match_efficientat import read_audio_window
from scripts.prepare_perception_av_temporal_matching import (
    _make_trial,
    build_manifest,
    choose_pair,
    event_rows,
)


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, axis: int):
        return FakeTensor(np.expand_dims(self.array, axis))

    def __getitem__(self, index):
        return FakeTensor(self.array[index])


class FakeTorch:
    @staticmethod
    def from_numpy(array: np.ndarray):
        return FakeTensor(array)


class FakeResample:
    @staticmethod
    def resample(tensor: FakeTensor, *, orig_freq: int, new_freq: int):
        count = round(tensor.shape[-1] * new_freq / orig_freq)
        source_x = np.arange(tensor.shape[-1], dtype=np.float64)
        target_x = np.linspace(0, tensor.shape[-1] - 1, count)
        return FakeTensor(np.interp(target_x, source_x, tensor.array[0])[None])


class FakeTorchaudio:
    functional = FakeResample()


def record(events: list[dict[str, object]]) -> dict[str, object]:
    return {
        "metadata": {
            "audio_samples": 48000 * 12,
            "audio_sample_rate": 48000,
            "num_frames": 30 * 12,
            "frame_rate": 30,
        },
        "sound_localisation": events,
    }


def event(event_id: int, label_id: int, start: float, end: float) -> dict[str, object]:
    return {
        "id": event_id,
        "label_id": label_id,
        "label": "Animal:sound",
        "timestamps": [int(start * 1_000_000), int(end * 1_000_000)],
    }


def test_event_windows_require_allowed_class_and_two_seconds_within_media() -> None:
    value = record(
        [
            event(1, 1, 0.1, 0.5),
            event(2, 1, 2.0, 2.5),
            event(3, 13, 5.0, 5.5),
            event(4, 1, 11.2, 11.8),
        ]
    )

    rows = event_rows(value)

    assert [row["event_id"] for row in rows] == [2]


def test_pair_requires_same_label_and_nonoverlapping_centered_windows() -> None:
    value = record(
        [
            event(1, 1, 2.0, 2.4),
            event(2, 1, 4.5, 4.9),
            event(3, 7, 8.0, 8.4),
        ]
    )

    pair = choose_pair("video_1", value, anchor_event_id=1, split="train")

    assert pair is not None
    assert pair["anchor"]["event_id"] == 1
    assert pair["partner"]["event_id"] == 2
    assert pair["center_gap_s"] >= 2.25


def test_pair_selection_and_option_order_are_deterministic() -> None:
    value = record(
        [
            event(1, 1, 2.0, 2.4),
            event(2, 1, 5.0, 5.4),
            event(3, 1, 8.0, 8.4),
        ]
    )

    first = choose_pair("video_1", value, split="valid")
    second = choose_pair("video_1", value, split="valid")
    assert first == second
    assert first is not None

    trial_a = _make_trial(first, split="valid")
    trial_b = _make_trial(second, split="valid")
    assert trial_a == trial_b
    assert len(trial_a["options"]) == 2
    assert trial_a["options"][trial_a["target_index"]]["role"] == "synchronized"


def test_other_split_pair_has_one_group_id_and_same_video_negative() -> None:
    value = record([event(1, 1, 2.0, 2.4), event(2, 1, 5.0, 5.4)])
    pair = choose_pair("video_42", value, split="train")
    assert pair is not None

    trial = _make_trial(pair, split="train")

    assert trial["group_id"] == "video_42"
    assert trial["source_video_id"] == "video_42"
    assert trial["same_label_hard_negative"] is True
    assert trial["same_video_hard_negative"] is True
    assert trial["target_index"] in (0, 1)


def test_audio_window_reader_extracts_exact_two_second_crop(tmp_path: Path) -> None:
    path = tmp_path / "small.wav"
    samples = np.arange(12 * 48_000, dtype="<i4")
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(4)
        sink.setframerate(48_000)
        sink.writeframes(samples.tobytes())

    waveform, metadata = read_audio_window(path, [1.0, 3.0], FakeTorchaudio, FakeTorch)

    assert waveform.shape == (1, 64_000)
    assert metadata["source_sample_range"] == [48_000, 144_000]
    assert metadata["resampled_sample_count"] == 64_000


def test_audio_window_reader_rejects_crop_outside_media(tmp_path: Path) -> None:
    path = tmp_path / "small.wav"
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(4)
        sink.setframerate(48_000)
        sink.writeframes(np.zeros(12 * 48_000, dtype="<i4").tobytes())

    with pytest.raises(ValueError, match="outside media duration"):
        read_audio_window(path, [11.0, 13.0], FakeTorchaudio, FakeTorch)


def test_manifest_builder_is_deterministic_and_groups_by_video(tmp_path: Path) -> None:
    def annotation(video_id: str, first_event: int) -> dict[str, object]:
        return {
            video_id: {
                "metadata": {
                    "audio_samples": 12 * 48_000,
                    "audio_sample_rate": 48_000,
                    "num_frames": 30 * 12,
                    "frame_rate": 30,
                },
                "sound_localisation": [
                    event(first_event, 1, 2.0, 2.4),
                    event(first_event + 1, 1, 5.0, 5.4),
                ],
            }
        }

    def write_archive(path: Path, member: str, records: dict[str, object]) -> None:
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(member, json.dumps(records))

    train_path = tmp_path / "train.zip"
    valid_path = tmp_path / "valid.zip"
    write_archive(train_path, "all_train.json", annotation("train_video", 1))
    write_archive(valid_path, "all_valid.json", annotation("valid_video", 3))
    cohort_path = tmp_path / "fixed.json"
    cohort_path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [{"split": "train", "video_id": "train_video", "event_id": 1}],
                    "valid": [{"split": "valid", "video_id": "valid_video", "event_id": 3}],
                }
            }
        ),
        encoding="utf-8",
    )
    fresh_path = tmp_path / "fresh-valid.json"
    fresh_path.write_text(json.dumps({"events": []}), encoding="utf-8")

    kwargs = {
        "train_annotations_path": train_path,
        "valid_annotations_path": valid_path,
        "fixed_cohort_path": cohort_path,
        "new_valid_selection_path": fresh_path,
        "added_train_videos": 0,
    }
    first = build_manifest(**kwargs)
    second = build_manifest(**kwargs)

    assert first == second
    assert len(first["train"]) == 1
    assert len(first["validation"]) == 1
    assert first["train"][0]["group_id"] == "train_video"
    assert first["validation"][0]["group_id"] == "valid_video"
    assert first["summary"]["train_validation_video_overlap"] == 0
    assert first["summary"]["official_test_split_opened"] is False


def test_manifest_builder_rejects_video_group_overlap(tmp_path: Path) -> None:
    def record(video_id: str, start_id: int) -> dict[str, object]:
        return {
            video_id: {
                "metadata": {
                    "audio_samples": 12 * 48_000,
                    "audio_sample_rate": 48_000,
                    "num_frames": 30 * 12,
                    "frame_rate": 30,
                },
                "sound_localisation": [
                    event(start_id, 1, 2.0, 2.4),
                    event(start_id + 1, 1, 5.0, 5.4),
                ],
            }
        }

    train_path, valid_path = tmp_path / "train.zip", tmp_path / "valid.zip"
    with ZipFile(train_path, "w") as archive:
        archive.writestr("all_train.json", json.dumps(record("same_video", 1)))
    with ZipFile(valid_path, "w") as archive:
        archive.writestr("all_valid.json", json.dumps(record("same_video", 3)))
    cohort_path = tmp_path / "fixed.json"
    cohort_path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [{"split": "train", "video_id": "same_video", "event_id": 1}],
                    "valid": [{"split": "valid", "video_id": "same_video", "event_id": 3}],
                }
            }
        ),
        encoding="utf-8",
    )
    fresh_path = tmp_path / "fresh.json"
    fresh_path.write_text(json.dumps({"events": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="video identity overlap"):
        build_manifest(
            train_annotations_path=train_path,
            valid_annotations_path=valid_path,
            fixed_cohort_path=cohort_path,
            new_valid_selection_path=fresh_path,
            added_train_videos=0,
        )
