from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrained_reuse.assemble_haptile_decision_features import (
    _state_split_proven,
    align_split_features,
    apply_video_window_eligibility,
)


def test_train_state_provenance_accepts_train_only_record_without_split_field() -> None:
    assert _state_split_proven({
        "status": "complete_train_only_state_text_extraction",
        "validation_or_sealed_audit_opened": False,
    }, "train")


def test_train_state_provenance_rejects_ambiguous_or_development_record() -> None:
    assert not _state_split_proven({
        "status": "complete_train_only_state_text_extraction",
    }, "train")
    assert not _state_split_proven({
        "status": "complete_train_only_state_text_extraction",
        "validation_or_sealed_audit_opened": False,
    }, "validation")


def test_video_eligibility_accepts_already_filtered_state_and_text_features() -> None:
    actions, state, video, text, videos, states = _fixture_inputs()
    eligible_path = actions[0]["path"]
    excluded_path = actions[1]["path"]
    eligible_video = next(row for row in video if row["path"] == eligible_path)
    selected_state = next(i for i, row in enumerate(state) if row["path"] == eligible_path)
    selected_text = next(
        i for i, path in enumerate(text["sample_paths"]) if path == eligible_path
    )
    selected_video = next(i for i, row in enumerate(video) if row["path"] == eligible_path)
    eligibility = {
        "format": "haptile_train_video_window_eligibility_v2",
        "split": "train",
        "input_receipt_sha256": "receipt",
        "action_preflight_sha256": "actions",
        "split_manifest_sha256": "manifest",
        "selection_uses_labels": False,
        "sealed_audit_opened": False,
        "criteria": {
            "context_seconds": 1.0,
            "frames_per_stream": 8,
            "max_frame_time_error_seconds": 0.1,
            "causal_cutoff_inclusive": True,
            "streams": [
                "/front-camera", "/wrist-camera", "/tactile-left", "/tactile-right"
            ],
            "excluded_only_for": "insufficient distinct causal frames in fixed window",
        },
        "eligible_samples": [{
            "path": eligible_path,
            "sha256": actions[0]["sha256"],
            "task": actions[0]["task"],
            "query_time_log_seconds": actions[0]["query_time_log_seconds"],
            "selected_frame_times_log_seconds": eligible_video[
                "selected_frame_times_log_seconds"
            ],
        }],
        "excluded_samples": [{"path": excluded_path, "sha256": actions[1]["sha256"]}],
        "eligible_examples": 1,
        "excluded_examples": 1,
        "corpus_examples": 2,
    }
    filtered_state = [state[selected_state]]
    filtered_text = {
        key: (value[[selected_text]] if isinstance(value, np.ndarray)
              and value.ndim > 0 and value.shape[0] == 2 else value)
        for key, value in text.items()
    }
    result = apply_video_window_eligibility(
        action_rows=actions,
        state_rows=filtered_state,
        video_rows=[eligible_video],
        text_cache=filtered_text,
        video_values=videos[[selected_video]],
        state_values=states[[selected_state]],
        eligibility=eligibility,
        split_name="train",
        receipt_sha256="receipt",
        action_sha256="actions",
        split_manifest_sha256="manifest",
    )
    assert [row["path"] for row in result[1]] == [eligible_path]
    assert result[4].shape[0] == result[5].shape[0] == 1
    assert result[3]["sample_paths"].tolist() == [eligible_path]


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
            "query_time_log_seconds": row["query_time_log_seconds"],
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
