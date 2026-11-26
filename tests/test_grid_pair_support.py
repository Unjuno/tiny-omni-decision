import numpy as np
import pytest

from tiny_omni_decision import grid_pair_support as module
from tiny_omni_decision.grid_role_supervision import pair_role_counts
from tiny_omni_decision.mask_pair_association import labelled_mask_pairs


def test_histograms_match_explicit_pairs_including_unknown_ambiguity_overlap():
    labels = np.array([[4, 6, -1, -2], [4, 5, -2, -1], [6, 5, 4, -1]], np.int64)
    windows = [(0, 1, 0), (1, 2, 0), (0, 2, 1)]
    pairs, _, _, ledger = labelled_mask_pairs(
        labels.ravel(), np.arange(4, dtype=np.int64) * 4, windows
    )
    roles = pair_role_counts(labels.ravel(), pairs, hidden=6, group=[4, 5], cover=4)
    actual = module.grid_pair_support(labels, windows, hidden=6, group=[4, 5], cover=4)
    assert actual["ledger"] == ledger and actual["roles"] == roles
    assert actual["roles"]["same_hidden"] == 1


def test_zero_support_is_kept_in_complete_denominators():
    labels = np.array([[-1, -2], [-1, -1]], np.int64)
    result = module.grid_pair_support(labels, [(0, 1, 0)], hidden=6, group=[4, 5], cover=4)
    assert result["ledger"]["all_pairs"] == 4 and result["ledger"]["labelled_pairs"] == 0
    assert result["roles"]["total"] == 0 and result["roles"]["same_hidden"] == 0


@pytest.mark.parametrize("bad", ["unknown_id", "float", "window"])
def test_invalid_namespace_or_array_or_time_routing_rejected(bad):
    labels = np.array([[4, 6], [4, 5]], np.int64)
    windows = [(0, 1, 0)]
    if bad == "unknown_id":
        labels[0, 0] = 99
    elif bad == "float":
        labels = labels.astype(float)
    else:
        windows = [(1, 0, 0)]
    with pytest.raises(ValueError):
        module.grid_pair_support(labels, windows, hidden=6, group=[4, 5], cover=4)
