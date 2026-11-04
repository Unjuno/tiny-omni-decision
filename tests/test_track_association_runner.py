import numpy as np

from scripts.diagnose_track_association import run_matches


def test_all_query_predictions_precede_posthoc_labels_and_misalignment_is_fixed():
    actual = np.broadcast_to(np.eye(3, dtype=np.float32)[:, None], (3, 2, 3)).copy()
    static = actual.copy()
    results = run_matches(actual, static)
    expected = np.broadcast_to(np.arange(3)[:, None], (3, 2))
    np.testing.assert_array_equal(results["actual"]["indices"], expected)
    np.testing.assert_array_equal(results["static"]["indices"], expected)
    np.testing.assert_array_equal(results["fixed_index"]["indices"], expected)
    np.testing.assert_array_equal(results["misaligned"]["indices"], (expected + 64) % 3)
    assert results["actual"]["indices"].shape == (3, 2)
    assert results["actual"]["scores"].dtype == np.float64
