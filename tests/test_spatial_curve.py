import pytest

from tiny_omni_decision.spatial_curve import require_prefix


def test_prefix_requires_eight_identical_records():
    old = [{"epoch": i, "nll": 1 / i} for i in range(1, 9)]
    require_prefix(old.copy(), old)
    with pytest.raises(ValueError):
        require_prefix(old[:-1], old)
    with pytest.raises(ValueError):
        require_prefix(old[:-1] + [{"epoch": 8, "nll": 0.2}], old)
