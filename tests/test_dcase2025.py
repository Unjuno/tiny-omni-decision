from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from tiny_omni_decision.dcase2025 import (
    assert_dcase2025_room_disjoint,
    load_dcase2025_devtrain_queries,
    parse_dcase2025_clip_member,
    queries_from_annotations,
    select_one_query_per_clip,
)
from tiny_omni_decision.dcase2025_media import stereo_from_foa, uniform_clip_frame_indices


def _clip(name: str = "fold3_room22_mix007_deg216_start0139"):
    parsed = parse_dcase2025_clip_member(
        f"metadata_dev/dev-train-sony/{name}.csv"
    )
    assert parsed is not None
    return parsed


def test_clip_member_parser_uses_room_as_split_identity() -> None:
    clip = _clip()
    assert clip.split == "train"
    assert clip.room_id == "sony:room22"
    assert clip.parent_recording == "fold3_room22_mix007"
    assert clip.view_degrees == 216
    assert clip.start_frame == 139
    assert parse_dcase2025_clip_member("metadata_dev/dev-train-sony/") is None
    assert parse_dcase2025_clip_member("metadata_dev/dev-test-sony/not-a-clip.csv") is None


def test_queries_reject_same_class_overlapping_sources() -> None:
    rows = [
        {"frame": 0, "class": 2, "source": 0, "azimuth": 20, "distance": 1, "onscreen": 1},
        {"frame": 1, "class": 2, "source": 0, "azimuth": 20, "distance": 1, "onscreen": 1},
        {"frame": 1, "class": 2, "source": 1, "azimuth": 40, "distance": 2, "onscreen": 0},
        {"frame": 2, "class": 2, "source": 0, "azimuth": 20, "distance": 1, "onscreen": 1},
        {"frame": 4, "class": 2, "source": 0, "azimuth": 20, "distance": 1, "onscreen": 1},
        {"frame": 5, "class": 2, "source": 0, "azimuth": 20, "distance": 1, "onscreen": 1},
        {"frame": 6, "class": 2, "source": 0, "azimuth": 20, "distance": 1, "onscreen": 1},
    ]
    queries, counts = queries_from_annotations(_clip(), rows)
    assert [item.event_frame for item in queries] == [5]
    assert counts == {
        "annotation_rows": 7,
        "unambiguous_segments": 1,
        "ambiguous_segments": 2,
    }


def test_queries_reject_malformed_and_out_of_range_annotations() -> None:
    rows = [{"frame": 50, "class": 2, "source": 0, "azimuth": 0, "distance": 1, "onscreen": 1}]
    with pytest.raises(ValueError, match="outside"):
        queries_from_annotations(_clip(), rows)
    with pytest.raises(ValueError, match="columns"):
        queries_from_annotations(_clip(), [{"frame": 0}])


def test_one_query_per_clip_is_stable_and_seeded() -> None:
    clip = _clip()
    rows = [
        {
            "frame": frame,
            "class": event_class,
            "source": 0,
            "azimuth": 0,
            "distance": 1,
            "onscreen": frame % 2,
        }
        for event_class in (1, 2)
        for frame in (0, 1)
    ]
    queries, _ = queries_from_annotations(clip, rows)
    first = select_one_query_per_clip(queries, seed=17)
    assert first == select_one_query_per_clip(reversed(queries), seed=17)
    assert len(first) == 1
    assert first[0] in queries


def test_development_loader_never_opens_devtest_annotations() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            "metadata_dev/dev-train-sony/fold3_room22_mix007_deg216_start0139.csv",
            "frame,class,source,azimuth,distance,onscreen\n0,2,0,20,1,1\n",
        )
        archive.writestr(
            "metadata_dev/dev-test-sony/fold3_room23_mix008_deg000_start0001.csv",
            "THIS LABEL CONTENT MUST NOT BE READ",
        )
    archive_buffer.seek(0)
    with zipfile.ZipFile(archive_buffer) as archive:
        names = archive.namelist()
        queries, inventory = load_dcase2025_devtrain_queries(archive)
        assert any("dev-test" in name for name in names)
        assert len(queries) == 1
        assert inventory["opened_devtest_annotation_members"] is False
        assert queries[0].split == "train"


def test_room_disjoint_gate_rejects_room_parent_and_clip_overlap() -> None:
    train = [_clip()]
    same_room = parse_dcase2025_clip_member(
        "metadata_dev/dev-test-sony/fold3_room22_mix009_deg000_start0001.csv"
    )
    assert same_room is not None
    with pytest.raises(ValueError, match="room_id"):
        assert_dcase2025_room_disjoint(train, [same_room])


def test_uniform_frame_indices_match_official_clip_timebase() -> None:
    indices = uniform_clip_frame_indices(start_frame=13, fps=29.97)
    assert indices == (39, 60, 82, 103, 124, 145, 167, 188)
    centered = uniform_clip_frame_indices(
        start_frame=13, fps=29.97, event_frame=25
    )
    assert centered == (104, 107, 110, 113, 115, 118, 121, 124)


def test_stereo_projection_matches_official_rotation_and_clipping_gate() -> None:
    foa = np.zeros((3, 4), dtype=np.float32)
    foa[:, 0] = 0.2
    foa[:, 1] = 0.1
    foa[:, 3] = 0.3
    stereo = stereo_from_foa(foa, view_degrees=90)
    assert stereo.shape == (3, 2)
    assert stereo[0].tolist() == pytest.approx([0.5, -0.1])
    foa[:, 0] = 0.8
    with pytest.raises(ValueError, match="clipped"):
        stereo_from_foa(foa, view_degrees=90)

