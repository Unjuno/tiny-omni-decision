from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark_path_a_latency_distribution import _choose_videos, _distribution


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_latency_distribution_uses_nearest_rank_percentile() -> None:
    summary = _distribution([8.0, 1.0, 4.0, 2.0, 6.0, 3.0, 7.0, 5.0])

    assert summary == {
        "count": 8,
        "min": 1.0,
        "median": 4.5,
        "p90_nearest_rank": 8.0,
        "max": 8.0,
    }
    assert _distribution([]) == {"count": 0}


def test_video_selection_uses_unique_scene_assets_and_stable_first_rows(
    tmp_path: Path,
) -> None:
    rows = []
    predictions = []
    scene_hashes = []
    for scene_number in (1, 2):
        media_path = Path("media/validation") / f"scene-{scene_number}.mp4"
        absolute_media = (
            tmp_path
            / "clevrer-expanded-seed17-256train-200val"
            / media_path
        )
        absolute_media.parent.mkdir(parents=True, exist_ok=True)
        content = f"fake-video-{scene_number}".encode()
        absolute_media.write_bytes(content)
        media_sha = hashlib.sha256(content).hexdigest()
        scene_hashes.append(media_sha)
        for choice in range(2):
            sample_id = f"scene-{scene_number}-choice-{choice}"
            rows.append(
                {
                    "id": sample_id,
                    "media_path": media_path.as_posix(),
                    "media_sha256": media_sha,
                    "scene_group_id": f"clevrer:validation:{scene_number}",
                    "question": "Which candidate is correct?",
                    "source_question": "Which event happened?",
                    "candidate_statement": f"candidate {choice}",
                }
            )
            predictions.append(
                {
                    "id": sample_id,
                    "prediction_index": choice,
                    "probabilities": [0.6, 0.4] if choice == 0 else [0.2, 0.8],
                }
            )

    _write_jsonl(
        tmp_path / "clevrer-binary-reasoning-v0" / "validation-binary-decisions.jsonl",
        rows,
    )
    _write_jsonl(
        tmp_path / "clevrer-binary-reasoning-v0" / "validation-predictions.jsonl",
        predictions,
    )

    selected = _choose_videos(tmp_path, 2)

    assert [spec["asset_id"] for spec in selected] == scene_hashes
    assert [spec["reference"]["sample_id"] for spec in selected] == [
        "scene-1-choice-0",
        "scene-2-choice-0",
    ]
    assert [spec["warm_reference"]["sample_id"] for spec in selected] == [
        "scene-1-choice-1",
        "scene-2-choice-1",
    ]
    assert all(
        spec["payload"]["expected_media_sha256"] == spec["asset_id"] for spec in selected
    )


def test_video_selection_fails_closed_when_distinct_scenes_are_insufficient(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clevrer-expanded-seed17-256train-200val" / "media/validation/v.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    media_sha = hashlib.sha256(b"video").hexdigest()
    rows = [
        {
            "id": f"q{choice}",
            "media_path": "media/validation/v.mp4",
            "media_sha256": media_sha,
            "scene_group_id": "scene-1",
            "question": "question",
            "source_question": "question",
            "candidate_statement": f"candidate-{choice}",
        }
        for choice in range(2)
    ]
    predictions = [
        {"id": f"q{choice}", "prediction_index": choice, "probabilities": [0.5, 0.5]}
        for choice in range(2)
    ]
    _write_jsonl(
        tmp_path / "clevrer-binary-reasoning-v0" / "validation-binary-decisions.jsonl",
        rows,
    )
    _write_jsonl(
        tmp_path / "clevrer-binary-reasoning-v0" / "validation-predictions.jsonl",
        predictions,
    )

    with pytest.raises(ValueError, match="requested 2 Video scenes but only found 1"):
        _choose_videos(tmp_path, 2)
