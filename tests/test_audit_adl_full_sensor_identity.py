import numpy as np
import pytest

from scripts.audit_adl_full_sensor_identity import (
    compare_sensor_metadata,
    metadata_map,
    receipt_archive_hashes,
    sensor_index,
)
from scripts.download_adl_archives import PRIOR_ARCHIVES


def test_sensor_index_accepts_only_named_sensor_members():
    assert sensor_index("17_stop-toaster_multisensor-5.hdf5") == 5
    with pytest.raises(ValueError):
        sensor_index("17_stop-toaster_multisensor-6.hdf5")
    with pytest.raises(ValueError):
        sensor_index("17_stop-toaster.hdf5")


def test_metadata_map_keeps_identity_and_validates_rates():
    result = metadata_map(
        np.array([1, 2]),
        np.array([10, 20]),
        np.array([100.0, 200.0]),
        np.array([16000.0, 16000.0]),
    )
    assert result == {(1, 10): (100.0, 16000.0), (2, 20): (200.0, 16000.0)}
    with pytest.raises(ValueError, match="duplicate"):
        metadata_map(
            np.array([1, 1]), np.array([10, 10]), np.array([1.0, 2.0]), np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="sample-rate"):
        metadata_map(np.array([1]), np.array([10]), np.array([1.0]), np.array([0.0]))


def test_compare_sensor_metadata_requires_exact_identity_and_reports_deltas():
    ref = {(1, 2): (3.0, 10.0)}
    other = {(1, 2): (3.25, 10.5)}
    assert compare_sensor_metadata(ref, other) == {
        "rows": 1,
        "max_abs_timestamp_delta_native_units": 0.25,
        "max_abs_sample_rate_delta": 0.5,
    }
    with pytest.raises(ValueError, match="identities differ"):
        compare_sensor_metadata(ref, {(1, 3): (3.0, 10.0)})


def test_receipt_archive_hashes_requires_all_pinned_prior_archives_and_sha256():
    prior = {key: {"sha256": "a" * 64} for key in PRIOR_ARCHIVES}
    receipt = {
        "downloads": [{"key": "17_stop-toaster.zip", "sha256": "b" * 64}],
        "prior_verified_archives": prior,
    }
    assert receipt_archive_hashes(receipt) == {
        "17_stop-toaster.zip": "b" * 64,
        **{key: "a" * 64 for key in PRIOR_ARCHIVES},
    }
    del prior[next(iter(prior))]
    with pytest.raises(ValueError, match="set mismatch"):
        receipt_archive_hashes(receipt)
