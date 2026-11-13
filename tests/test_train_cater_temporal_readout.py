from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.train_cater_temporal_readout import (  # noqa: E402
    _batch_metrics,
    _cluster_bootstrap_accuracy_delta,
    _feature_for_mode,
    feature_record_for_example,
    group_by_video,
    prepare_examples,
)


def _row(sample_id: str, video_id: str, interval: list[int], target: str = "a"):
    return {
        "id": sample_id,
        "video_id": video_id,
        "split": "train",
        "question": f"which object moves from {interval[0]} to {interval[1]}?",
        "frame_interval": interval,
        "options": [{"id": "a", "text": "red cube"}, {"id": "b", "text": "blue sphere"}],
        "target_option_id": target,
    }


def test_prepare_examples_uses_manifest_target_without_global_video_offsets() -> None:
    row = _row("q1", "video1", [75, 150], target="b")
    embeddings = {
        row["question"]: torch.zeros(384),
        "red cube": torch.ones(384),
        "blue sphere": torch.full((384,), 2.0),
    }

    example = prepare_examples([row], embeddings)[0]

    assert example["target"] == 1
    assert "start_frame" not in example
    assert "end_frame" not in example
    assert example["options"].shape == (2, 384)


def test_group_by_video_keeps_all_intervals_in_one_group() -> None:
    rows = [_row("q1", "video1", [0, 75]), _row("q2", "video1", [75, 150])]
    embeddings = {
        text: torch.zeros(384)
        for row in rows
        for text in [row["question"], *(option["text"] for option in row["options"])]
    }

    groups = group_by_video(prepare_examples(rows, embeddings))

    assert len(groups) == 1
    assert groups[0][0] == "video1"
    assert len(groups[0][1]) == 2


def test_video_permutation_resolves_the_same_local_task_interval() -> None:
    example = {"row": _row("q1", "original", [75, 150])}
    sample_record = {"feature_path": "train/q1.npz"}
    same_interval_record = {"feature_path": "train/other-75-150.npz"}

    selected = feature_record_for_example(
        {("validation", "q1"): sample_record},
        {("validation", "replacement", (75, 150)): same_interval_record},
        "validation",
        example,
        video_id="replacement",
    )

    assert selected is same_interval_record


def test_feature_controls_keep_expected_temporal_content() -> None:
    video = torch.arange(13 * 2 * 3, dtype=torch.float32).reshape(13, 2, 3)

    no_video = _feature_for_mode(video, "text_only", 17, (0, 12))
    static = _feature_for_mode(video, "static_first", 17, (0, 12))
    reversed_video = _feature_for_mode(video, "reverse_interval", 17, (0, 12))

    assert torch.count_nonzero(no_video) == 0
    assert torch.equal(static[0], static[-1])
    assert torch.equal(reversed_video[0], video[-1])
    assert torch.equal(reversed_video[-1], video[0])


def test_interval_metrics_are_macro_averaged_and_group_bootstrap_is_paired() -> None:
    video, text = [], []
    for group in ("g1", "g2", "g3"):
        for start in (0, 75, 150, 225):
            sample_id = f"{group}-{start}"
            base = {
                "id": sample_id,
                "video_id": group,
                "interval": [start, start + 75],
                "options": ["a", "b"],
                "target_index": 0,
            }
            video.append({**base, "probabilities": [0.8, 0.2], "correct": True})
            text.append({**base, "probabilities": [0.4, 0.6], "correct": False})

    metrics = _batch_metrics(video)
    paired = _cluster_bootstrap_accuracy_delta(video, text, repetitions=50)

    assert metrics["sample_weighted"]["count"] == 12
    assert metrics["macro_by_interval"]["accuracy"] == 1.0
    assert paired["macro_accuracy_delta_video_minus_text_only"] == 1.0
    assert paired["scene_groups"] == 3
