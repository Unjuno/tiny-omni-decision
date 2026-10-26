import pytest

from scripts.preflight_starss23_spatial_decision import (
    azimuth_sector,
    canonical_class,
    interval_sector_targets,
)


@pytest.mark.parametrize(
    ("angle", "sector"),
    [
        (0, 0),
        (45, 1),
        (90, 2),
        (135, 3),
        (180, 4),
        (-180, 4),
        (-135, 5),
        (-90, 6),
        (-45, 7),
        (22, 0),
        (23, 1),
        (-22, 0),
        (-23, 7),
    ],
)
def test_azimuth_maps_to_eight_front_referenced_sectors(angle, sector):
    assert azimuth_sector(angle) == sector


@pytest.mark.parametrize("angle", [-181, 181, 360, -360])
def test_azimuth_rejects_angles_outside_source_contract(angle):
    with pytest.raises(ValueError, match=r"\[-180, 180\]"):
        azimuth_sector(angle)


def test_speech_gender_labels_share_one_event_class():
    assert canonical_class(0) == canonical_class(1) == 0
    assert canonical_class(2) == 2


@pytest.mark.parametrize("value", [-1, 13])
def test_class_rejects_unknown_source_labels(value):
    with pytest.raises(ValueError, match=r"\[0, 12\]"):
        canonical_class(value)


def test_interval_targets_merge_speech_and_keep_concurrent_sectors():
    rows = [
        [0, 0, 1, 0, 0, 100],
        [49, 1, 2, 90, 0, 100],
        [50, 8, 0, -90, 0, 200],
    ]

    targets = interval_sector_targets(rows, start_tick=0, stop_tick=50)

    assert len(targets) == 12 * 8
    assert targets[0 * 8 + 0] == 1
    assert targets[0 * 8 + 2] == 1
    assert targets[7 * 8 + 6] == 0
    assert sum(targets) == 2


@pytest.mark.parametrize("interval", [(0, 0), (-1, 3), (2, 1)])
def test_interval_target_builder_rejects_invalid_bounds(interval):
    with pytest.raises(ValueError, match="non-empty"):
        interval_sector_targets([], *interval)
