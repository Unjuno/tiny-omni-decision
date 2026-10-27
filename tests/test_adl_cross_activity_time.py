import zipfile

import pytest

from scripts.audit_adl_cross_activity_time import compare, ordered_hdf5_members


def test_zip_members_order_by_filename():
    members = [zipfile.ZipInfo("z.hdf5"), zipfile.ZipInfo("a.hdf5"), zipfile.ZipInfo("other")]
    assert [member.filename for member in ordered_hdf5_members(members)] == ["a.hdf5", "z.hdf5"]


def test_compare_counts_validation_timestamp_collisions_and_gaps():
    result = compare([1.0, 10.0], [1.0, 9.0, 20.0])
    assert result["equal_timestamp_validation_events"] == 1
    assert result["nearest_gap_quantiles_native_units"]["0"] == 0.0
    assert result["nearest_gap_quantiles_native_units"]["1"] == 10.0
    assert result["validation_events"] == 3


@pytest.mark.parametrize("train,validation", [([], [1]), ([1], []), ([1], [float("nan")])])
def test_compare_rejects_invalid_times(train, validation):
    with pytest.raises(ValueError):
        compare(train, validation)
