import pytest

from scripts.compare_native_capacity_control import select_curve_points


def test_curve_comparison_reports_only_shared_saved_points():
    scheduled = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048}
    common, reported, missing = select_curve_points(
        {
            "baseline": scheduled,
            "lstm": scheduled | {2304},
            "capacity_mlp": scheduled,
        }
    )

    assert common == sorted(scheduled)
    assert reported == [512, 768, 1024, 1280, 1536, 1792, 2048]
    assert missing == [1920]


def test_curve_comparison_fails_when_required_common_point_is_missing():
    with pytest.raises(ValueError, match="required common learning-curve checkpoint"):
        select_curve_points(
            {
                "baseline": {0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048},
                "lstm": {0, 256, 512, 768, 1024, 1280, 1792, 2048},
            }
        )
