from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts import audit_avatar_rasterizer_parity as parity


def test_official_union_mask_matches_pinned_polygon_conversion_contract():
    calls = []

    def fill_poly(mask, polygons, color):
        points = polygons[0]
        calls.append((points.copy(), color))
        for x, y in points:
            mask[y, x] = color

    cv2 = SimpleNamespace(fillPoly=fill_poly)
    annotations = [
        {"segmentation": [[0.9, 0.9, 2.8, 1.2]]},
        {"segmentation": None},
        {"segmentation": [[1.9, 2.9]]},
    ]

    mask = parity._official_opencv_union_mask(
        4,
        4,
        annotations,
        cv2=cv2,
        np=np,
    )

    assert mask.dtype == np.uint8
    assert len(calls) == 2
    assert calls[0][0].tolist() == [[0, 0], [2, 1]]
    assert calls[1][0].tolist() == [[1, 2]]
    assert np.count_nonzero(mask) == 3


def test_compare_masks_counts_full_pixel_and_point_label_differences():
    pillow = Image.new("1", (4, 4), 0)
    pillow.putpixel((1, 1), 1)
    opencv = np.zeros((4, 4), dtype=np.uint8)
    opencv[1, 1] = 1
    opencv[3, 3] = 1

    different, total, grids = parity._compare_masks(pillow, opencv, (2,), np)

    assert (different, total) == (1, 16)
    assert grids["2"] == {
        "queries_per_frame": 4,
        "pillow_positive_points": 1,
        "opencv_positive_points": 2,
        "different_point_labels": 1,
    }


def test_center_samples_use_same_truncated_pixel_mapping_as_support_audit():
    xs, ys = parity._center_samples(width=640, height=360, side=2)

    assert xs == [160, 480]
    assert ys == [90, 270]
