import numpy as np
import pytest

from scripts.slot_feature_sampling import pack_slots, sample_indices


def test_samples_cover_full_interval():
    ids = sample_indices(186)
    assert len(ids) == 32 and ids[0] == 0 and ids[-1] == 185
    assert len(set(ids)) == 32


def test_short_clip_rejected_not_duplicated():
    with pytest.raises(ValueError):
        sample_indices(12)


def test_time_attached_to_each_slot():
    slots = np.ones((32, 7, 128), dtype=np.float32)
    x = pack_slots(slots, sample_indices(186), 186)
    assert x.shape == (224, 129)
    assert np.all(x[:7, -1] == 0) and np.all(x[-7:, -1] == 1)
    assert np.all(x[:, :128] == 1)
