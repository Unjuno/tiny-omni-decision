import numpy as np
import pytest

from scripts.extract_eidon_videoprism_features import extract_split, target_times


def test_target_times_are_uniform_and_centered_in_five_second_window():
    times = target_times(15.0)
    np.testing.assert_allclose(times, np.linspace(5.0, 10.0, 8))


@pytest.mark.parametrize("duration,count", [(0, 8), (4.9, 8), (10, 0), (10, True)])
def test_target_times_reject_invalid_window_inputs(duration, count):
    with pytest.raises(ValueError):
        target_times(duration, count=count)


def test_videoprism_extraction_refuses_sealed_audit_before_reading(tmp_path):
    with pytest.raises(ValueError, match="only train and development"):
        extract_split(tmp_path / "anything.json", tmp_path / "records.json",
                      tmp_path / "model.npz", tmp_path / "source", "sealed_audit",
                      tmp_path / "out")
