import pytest

from scripts.audit_physionpp_temporal_sampling_coverage import (
    contiguous_runs,
    coverage_counts,
    sample_indices,
)


def test_sample_indices_include_endpoints_at_fixed_spacing():
    assert sample_indices(100, 8) == (0, 14, 28, 42, 57, 71, 85, 99)


def test_sampling_coverage_reports_exact_and_tolerant_contact_capture():
    assert coverage_counts(100, [13, 14, 15], 8, tolerance=0) == (True, 1)
    assert coverage_counts(100, [13, 14, 15], 8, tolerance=1) == (True, 3)
    assert coverage_counts(100, [10, 11], 8, tolerance=1) == (False, 0)


def test_contiguous_contact_runs():
    assert contiguous_runs([]) == 0
    assert contiguous_runs([1, 2, 3, 7, 9, 10]) == 3


@pytest.mark.parametrize("indices", [[2, 1], [1, 1]])
def test_contiguous_runs_reject_unsorted_or_duplicate_indices(indices):
    with pytest.raises(ValueError, match="sorted and unique"):
        contiguous_runs(indices)


@pytest.mark.parametrize(("frame_count", "sample_count"), [(1, 8), (8, 1), (8, 9)])
def test_sample_indices_reject_incompatible_counts(frame_count, sample_count):
    with pytest.raises(ValueError, match="incompatible"):
        sample_indices(frame_count, sample_count)
