import numpy as np
import pytest

from scripts.smoke_robomnist_frozen_features import (
    audio_windows,
    nearest_frame_indices,
    select_runtime_refs,
)


def test_smoke_selection_uses_first_train_inputs_without_labels():
    rows = [
        {
            "id": "first",
            "target": 1,
            "model_input": {"question": "fixed", "options": ["A", "B"]},
            "source_refs": {
                "video": {"group_id": "g1", "member": "v.mp4"},
                "audio_candidates": [
                    {"group_id": "g2", "member": "a2.wav"},
                    {"group_id": "g1", "member": "a1.wav"},
                ],
            },
        }
    ]
    result = select_runtime_refs(rows)
    assert set(result) == {"video", "audio_candidates"}
    assert result["audio_candidates"][0]["group_id"] == "g2"
    with pytest.raises(ValueError, match="two"):
        select_runtime_refs(
            [{**rows[0], "source_refs": {**rows[0]["source_refs"], "audio_candidates": []}}]
        )


def test_video_frame_selection_uses_actual_pts_and_rejects_missing_coverage():
    assert nearest_frame_indices([0, 0.8, 1.3, 2.1], [1, 2]) == [1, 3]
    with pytest.raises(ValueError, match="coverage"):
        nearest_frame_indices([0, 1], [2])
    with pytest.raises(ValueError, match="increasing"):
        nearest_frame_indices([0, 1, 0.5], [0.5])


def test_audio_windows_keep_time_order_without_padding_or_wrapping():
    waveform = np.arange(80, dtype=np.float32)
    windows = audio_windows(waveform, 10, [1, 4, 7], 2)
    np.testing.assert_array_equal(windows[0], waveform[:20])
    np.testing.assert_array_equal(windows[1], waveform[30:50])
    np.testing.assert_array_equal(windows[2], waveform[60:80])
    with pytest.raises(ValueError, match="coverage"):
        audio_windows(waveform, 10, [7.5], 2)
