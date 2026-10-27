import pytest

from scripts.audit_source_temporal_coverage import summarize


def test_counts_variable_windows_frames_and_padding():
    rows = [
        {
            "audio_windows": [
                {
                    "source_start_us": 0,
                    "source_end_us": 1999999,
                    "pad_left_us": 0,
                    "pad_right_us": 1,
                }
            ],
            "candidates": [{"candidate_id": "a", "frames": [{"frame_id": 2}]}],
        }
    ]
    result = summarize(rows)
    assert result["audio_windows_per_record"] == {1: 1}
    assert result["video_frames_per_candidate"] == {1: 1}
    assert result["records_with_audio_padding"] == 1


def test_rejects_duplicate_frames():
    rows = [
        {
            "audio_windows": [
                {
                    "source_start_us": 0,
                    "source_end_us": 2000000,
                    "pad_left_us": 0,
                    "pad_right_us": 0,
                }
            ],
            "candidates": [{"candidate_id": "a", "frames": [{"frame_id": 1}, {"frame_id": 1}]}],
        }
    ]
    with pytest.raises(ValueError):
        summarize(rows)
