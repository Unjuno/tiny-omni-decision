from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrained_reuse.assemble_haptile_decision_features import align_split_features


def _fixture_inputs():
    actions = [
        {"path": "data/task_a/ep_a.mcap", "sha256": "hash-a", "target": "-x",
         "task": "task_a", "query_time_log_seconds": 10.0},
        {"path": "data/task_b/ep_b.mcap", "sha256": "hash-b", "target": "+x",
         "task": "task_b", "query_time_log_seconds": 20.0},
    ]
    state = [
        {"path": actions[1]["path"], "sha256": "hash-b", "target": "+x",
         "task_group": "task_b"},
        {"path": actions[0]["path"], "sha256": "hash-a", "target": "-x",
         "task_group": "task_a"},
    ]
    base_offsets = np.linspace(-1.0, 0.0, 8).tolist()
    video = []
    for row in reversed(actions):
        query = row["query_time_log_seconds"]
        offsets = [value for value in base_offsets]
        video.append({
            "path": row["path"], "sha256": row["sha256"],
            "target": row["target"], "task": row["task"],
            "selected_frame_times_log_seconds": {
                stream: [query + value for value in offsets]
                for stream in (
                    "/front-camera", "/wrist-camera", "/tactile-left", "/tactile-right"
                )
            },
        })
    text = {
        "sample_paths": np.asarray([actions[1]["path"], actions[0]["path"]]),
        "media_sha256": np.asarray(["hash-b", "hash-a"]),
        "targets": np.asarray(["+x", "-x"]),
        "context_features": np.stack([
            np.full(384, 200, dtype=np.float32), np.full(384, 100, dtype=np.float32)
        ]),
        "context_features_no_instruction": np.stack([
            np.full(384, 20, dtype=np.float32), np.full(384, 10, dtype=np.float32)
        ]),
        "option_features": np.stack([
            np.full((7, 1536), 2, dtype=np.float32),
            np.full((7, 1536), 1, dtype=np.float32),
        ]),
        "option_features_no_instruction": np.stack([
            np.full((7, 1536), 2, dtype=np.float32),
            np.full((7, 1536), 1, dtype=np.float32),
        ]),
    }
    videos = np.stack([
        np.full((4, 8, 768), 2, dtype=np.float32),
        np.full((4, 8, 768), 1, dtype=np.float32),
    ])
    states = np.stack([
        np.full((8, 25), 20, dtype=np.float32),
        np.full((8, 25), 10, dtype=np.float32),
    ])
    return actions, state, video, text, videos, states


def test_aligns_features_by_path_hash_and_target_not_row_order() -> None:
    args = _fixture_inputs()
    result = align_split_features(
        action_rows=args[0], state_rows=args[1], video_rows=args[2],
        text_cache=args[3], video_values=args[4], state_values=args[5],
    )
    assert result["sample_paths"].tolist() == [row["path"] for row in args[0]]
    np.testing.assert_array_equal(result["video_values"][:, 0, 0, 0], [1, 2])
    np.testing.assert_array_equal(result["state_values"][:, 0, 0], [10, 20])
    np.testing.assert_array_equal(result["context_features"][:, 0], [100, 200])
    np.testing.assert_array_equal(
        result["context_features_no_instruction"][:, 0], [10, 20]
    )
    np.testing.assert_array_equal(result["option_features"][:, 0, 0], [1, 2])
    np.testing.assert_array_equal(result["targets"], [0, 1])
    assert result["video_timestamps_ms"][0, 0] == pytest.approx(-1000)
    assert result["video_timestamps_ms"][0, -1] == pytest.approx(0)


def test_rejects_cross_source_media_hash_mismatch() -> None:
    args = _fixture_inputs()
    args[2][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="media hash or target differs"):
        align_split_features(
            action_rows=args[0], state_rows=args[1], video_rows=args[2],
            text_cache=args[3], video_values=args[4], state_values=args[5],
        )


def test_rejects_future_video_timestamp() -> None:
    args = _fixture_inputs()
    args[2][0]["selected_frame_times_log_seconds"]["/front-camera"][-1] += 0.001
    with pytest.raises(ValueError, match="noncausal"):
        align_split_features(
            action_rows=args[0], state_rows=args[1], video_rows=args[2],
            text_cache=args[3], video_values=args[4], state_values=args[5],
        )
