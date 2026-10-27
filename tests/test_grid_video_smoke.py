import pytest

pytest.importorskip('av')
from scripts.audit_grid_video_smoke import validate_times


def test_reject_non_monotonic_timestamps():
    with pytest.raises(ValueError, match='timestamps'):
        validate_times([0.0, 0.04, 0.04])


def test_accept_ordered_timestamps():
    assert len(validate_times([0.0, 0.04, 0.08])) == 3
