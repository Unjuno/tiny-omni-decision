from __future__ import annotations

import json

import pytest

from scripts import audit_avatar_point_grid_support as point_audit


def _frame(root, group: str, *, task: str = "Mixed-Sound", polygon=None) -> None:
    folder = root / f"{group}_00072"
    folder.mkdir(parents=True, exist_ok=True)
    annotations = [
        {
            "task": task,
            "audio_visual_category": "Guitar",
            "segmentation": [polygon] if polygon is not None else None,
        }
    ]
    (folder / "00001.json").write_text(
        json.dumps(
            {
                "video_id": f"{group}_72",
                "frame_number": 1,
                "original_width": 4,
                "original_height": 4,
                "fps": 1,
                "duration": 2,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )


def test_grid_support_uses_union_masks_and_global_train_groups_only(tmp_path, monkeypatch):
    train = tmp_path / "train"
    groups = ["abcdefghijk", "bcdefghijkl", "cdefghijklm"]
    exposed = "defghijklmn"
    left_top = [0, 0, 2, 0, 2, 2, 0, 2]
    bottom_right = [2, 2, 4, 2, 4, 4, 2, 4]
    for group in groups:
        _frame(train, group, polygon=left_top)
        # A second visible task annotation in the same frame contributes to
        # the official all-visible-mask union.
        path = train / f"{group}_00072" / "00001.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["annotations"].append(
            {
                "task": "Multi-Entity",
                "audio_visual_category": "Piano",
                "segmentation": [bottom_right],
            }
        )
        path.write_text(json.dumps(row), encoding="utf-8")
    _frame(train, exposed, polygon=[0, 0, 4, 0, 4, 4, 0, 4])

    monkeypatch.setattr(
        point_audit,
        "collect_prior_group_usage",
        lambda _root: ({exposed}, set(), {}, {}),
    )
    report = point_audit.audit_point_grid_support(
        train,
        artifact_root=tmp_path / "unused-artifacts",
        seed=17,
        validation_fraction=0.4,
        grid_sizes=(2, 4),
    )

    split = report["global_group_split"]
    support = report["mixed_sound_train_support"]
    grid2 = report["grid_center_support"]["2"]
    grid4 = report["grid_center_support"]["4"]
    assert split["unexposed_groups"] == 3
    assert split["train_groups"] + split["validation_groups"] == 3
    assert support["source_group_count"] == split["train_groups"]
    assert support["annotated_frame_count"] == split["train_groups"]
    assert support["frames_without_visible_union_mask"] == 0
    assert grid2["positive_queries"] == 2 * support["annotated_frame_count"]
    assert grid4["positive_queries"] == 12 * support["annotated_frame_count"]
    assert grid2["frames_with_multiple_positive_queries"] == support["annotated_frame_count"]
    assert grid4["frames_with_multiple_positive_queries"] == support["annotated_frame_count"]
    assert report["source_group_ids_emitted"] is False
    assert report["official_validation_annotations_accessed"] is False
    assert report["internal_validation_group_masks_or_dimensions_used"] is False
    assert exposed not in json.dumps(report)


def test_grid_support_requires_isolated_train_directory(tmp_path, monkeypatch):
    audit = tmp_path / "audit"
    audit.mkdir()
    monkeypatch.setattr(
        point_audit,
        "collect_prior_group_usage",
        lambda _root: (set(), set(), {}, {}),
    )
    with pytest.raises(ValueError, match="isolated TRAIN"):
        point_audit.audit_point_grid_support(audit, artifact_root=tmp_path)


def test_grid_support_rejects_malformed_polygons():
    with pytest.raises(ValueError, match="at least one xy point"):
        point_audit._polygons_for_visible_annotations([{"segmentation": [[0, 0, 1]]}])


def test_degenerate_line_and_point_masks_contribute_pixel_support():
    mask = point_audit._rasterize_union(4, 4, [[1, 1, 1, 2], [3, 3]])

    assert point_audit._grid_positive_count(mask, 2) == 2
