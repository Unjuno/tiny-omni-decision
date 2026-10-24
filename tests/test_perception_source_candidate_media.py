import json
import sys
from types import SimpleNamespace

import pytest

from scripts.preflight_perception_source_candidate_media import (
    audio_interval_coverage,
    describe_terminal_frame_repair,
    inspect_video_group,
    preflight,
    select_records_by_id,
    select_smoke_records,
    sha256_file,
)


def record(video_id, sound_id, *, none_visible=False, cut_frame=-1):
    return {
        "status": "candidate",
        "record_id": f"{video_id}:sound:{sound_id}",
        "group_id": video_id,
        "media_reference": {"video_id": video_id},
        "supervision": {"none_visible": none_visible},
        "provenance": {"upstream_cut_frame": cut_frame},
    }


def test_smoke_selection_is_seeded_group_disjoint_and_media_bounded():
    records = [record(f"v{i}", j) for i in range(5) for j in range(2)]
    media_ids = {f"v{i}" for i in range(4)}
    selected_a = select_smoke_records(records, media_ids, limit=3, seed=17)
    selected_b = select_smoke_records(records, media_ids, limit=3, seed=17)
    assert [item["record_id"] for item in selected_a] == [item["record_id"] for item in selected_b]
    assert len(selected_a) == 3
    assert len({item["group_id"] for item in selected_a}) == 3
    assert all(item["group_id"] in media_ids for item in selected_a)


def test_audio_interval_coverage_clips_and_unions_decoded_frames():
    coverage = audio_interval_coverage([100_000, 500_000], [(0.0, 0.3), (0.25, 0.6), (0.8, 1.0)])
    assert coverage == 0.4


def test_audio_interval_coverage_reports_missing_samples():
    coverage = audio_interval_coverage([0, 1_000_000], [(0.0, 0.4), (0.6, 1.0)])
    assert coverage == 0.8


def test_smoke_selection_can_require_none_visible_and_cut_frame_groups():
    records = [
        record("visible", 0),
        record("none", 0, none_visible=True),
        record("cut", 0, cut_frame=25),
        record("other", 0),
    ]
    selected = select_smoke_records(
        records,
        {"visible", "none", "cut", "other"},
        limit=3,
        seed=17,
        require_none_visible=1,
        require_cut_frame=1,
    )
    assert any(item["supervision"]["none_visible"] for item in selected)
    assert any(item["provenance"]["upstream_cut_frame"] >= 0 for item in selected)
    assert len({item["group_id"] for item in selected}) == 3


def test_explicit_record_selection_preserves_all_rows_and_requested_order():
    rows = [record("v1", 0), record("v1", 1), record("v2", 0)]
    selected = select_records_by_id(rows, {"v1", "v2"}, ["v1:sound:1", "v1:sound:0"])
    assert [item["record_id"] for item in selected] == ["v1:sound:1", "v1:sound:0"]
    assert [item["group_id"] for item in selected] == ["v1", "v1"]


def test_explicit_record_selection_fails_closed_for_missing_or_duplicate_ids():
    rows = [record("v1", 0)]
    with pytest.raises(ValueError, match="absent from local candidate media"):
        select_records_by_id(rows, {"v1"}, ["v2:sound:0"])
    with pytest.raises(ValueError, match="contains duplicates"):
        select_records_by_id(rows, {"v1"}, ["v1:sound:0", "v1:sound:0"])


def test_explicit_cohort_report_hashes_selection_jsonl(tmp_path, monkeypatch):
    rows = [record("v1", 0), record("v1", 1), record("v2", 0)]
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    clips_path = tmp_path / "clips.json"
    clips_path.write_text(
        json.dumps(
            {
                "clips": [
                    {"metadata": {"video_id": video_id, "split": "train"}}
                    for video_id in ("v1", "v2")
                ]
            }
        ),
        encoding="utf-8",
    )
    cohort_path = tmp_path / "development.jsonl"
    cohort_path.write_text(
        json.dumps(rows[0]) + "\n" + json.dumps(rows[1]) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "scripts.preflight_perception_source_candidate_media.inspect_video_group",
        lambda group, _clip, _media: [
            {
                "record_id": row["record_id"],
                "video_id": row["group_id"],
                "none_visible": False,
                "upstream_cut_frame": -1,
            }
            for row in group
        ],
    )

    report = preflight(
        corpus_path,
        clips_path,
        tmp_path,
        record_ids_path=cohort_path,
        progress_interval=2,
    )

    assert report["selection"] == "explicit_ordered_record_ids"
    assert report["selected_record_count"] == 2
    assert report["video_group_count"] == 1
    assert report["selection_source_jsonl_sha256"] == sha256_file(cohort_path)
    assert "selected_record_ids_sha256" not in report


def test_terminal_frame_repair_is_symmetric_and_keeps_each_candidate_usable():
    row = {
        "record_id": "v1:sound:1",
        "request": {
            "candidates": [
                {"candidate_id": "option_0", "frame_boxes": [{"frame_id": 0}, {"frame_id": 3}]},
                {"candidate_id": "option_1", "frame_boxes": [{"frame_id": 1}, {"frame_id": 3}]},
            ]
        },
    }
    repair = describe_terminal_frame_repair(
        row, missing_frame_ids=[3], declared_frame_count=4, decoded_frame_count=3
    )
    assert repair == {
        "record_id": "v1:sound:1",
        "frame_id": 3,
        "candidate_ids": ["option_0", "option_1"],
        "retained_frame_count_by_candidate": {"option_0": 1, "option_1": 1},
        "declared_frame_count": 4,
        "decoded_frame_count": 3,
    }


def test_terminal_frame_repair_allows_unaffected_candidates_and_rejects_empty_sequences():
    partial = {
        "record_id": "v1:sound:1",
        "request": {
            "candidates": [
                {"candidate_id": "option_0", "frame_boxes": [{"frame_id": 0}, {"frame_id": 3}]},
                {"candidate_id": "option_1", "frame_boxes": [{"frame_id": 1}]},
            ]
        },
    }
    repair = describe_terminal_frame_repair(
        partial, missing_frame_ids=[3], declared_frame_count=4, decoded_frame_count=3
    )
    assert repair["candidate_ids"] == ["option_0"]
    assert repair["retained_frame_count_by_candidate"] == {"option_0": 1, "option_1": 1}
    with pytest.raises(ValueError, match="no usable frames"):
        describe_terminal_frame_repair(
            {
                **partial,
                "request": {
                    "candidates": [{"candidate_id": "only", "frame_boxes": [{"frame_id": 3}]}]
                },
            },
            missing_frame_ids=[3],
            declared_frame_count=4,
            decoded_frame_count=3,
        )


def test_terminal_frame_repair_rejects_internal_gaps_and_multiple_missing_frames():
    row = {
        "record_id": "v1:sound:1",
        "request": {
            "candidates": [
                {"candidate_id": "option_0", "frame_boxes": [{"frame_id": 0}, {"frame_id": 3}]}
            ]
        },
    }
    with pytest.raises(ValueError, match="one missing declared terminal frame"):
        describe_terminal_frame_repair(
            row, missing_frame_ids=[2], declared_frame_count=4, decoded_frame_count=3
        )
    with pytest.raises(ValueError, match="one missing declared terminal frame"):
        describe_terminal_frame_repair(
            row, missing_frame_ids=[3, 4], declared_frame_count=5, decoded_frame_count=3
        )


def test_video_group_preflight_decodes_shared_media_once(tmp_path, monkeypatch):
    media = tmp_path / "v1.mp4"
    media.write_bytes(b"fixed-video")
    video_stream = object()
    audio_stream = object()

    class FakeContainer:
        streams = SimpleNamespace(video=[video_stream], audio=[audio_stream])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, stream):
            if stream is video_stream:
                return iter(
                    [
                        SimpleNamespace(time=0.0, width=16, height=16),
                            SimpleNamespace(time=0.5, width=16, height=16),
                    ]
                )
            return iter([SimpleNamespace(time=0.0, sample_rate=10, samples=10)])

    opens = []
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda path: (opens.append(path) or FakeContainer())),
    )
    clip = {
        "bytes": media.stat().st_size,
        "sha256": sha256_file(media),
        "cut_frame": -1,
        "metadata": {
            "video_id": "v1",
            "split": "train",
            "num_frames": 100,
            "frame_rate": 10,
        },
    }
    rows = []
    for index, frame_id in enumerate((0, 1)):
        row = record("v1", index)
        row["request"] = {
            "audio_interval_us": [index * 500_000, (index + 1) * 500_000],
            "candidates": [
                {
                    "candidate_id": f"option-{index}",
                    "frame_boxes": [{"frame_id": frame_id}],
                }
            ],
        }
        rows.append(row)

    reports = inspect_video_group(rows, clip, tmp_path)
    assert len(opens) == 2  # one video-stream and one audio-stream decode per video
    assert [row["record_id"] for row in reports] == [row["record_id"] for row in rows]
    assert [row["decoded_frame_ids"] for row in reports] == [[0], [1]]
    assert all(row["decoded_audio_coverage_s"] == pytest.approx(0.5) for row in reports)
