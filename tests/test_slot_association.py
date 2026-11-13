import numpy as np
import pytest

from scripts.associate_slot_masks import associate


def test_tracks_slot_permutation_without_later_labels():
    hard = np.zeros((3, 4, 4), int)
    hard[:, :, 2:] = 1
    hard[1] = 1 - hard[1]
    masks = np.eye(2)[hard].transpose(0, 3, 1, 2)
    assert associate(masks, [0, 30, 60], [0, 0, 0.5, 1]) == [0, 1, 0]


def test_rejects_unordered_frames():
    with pytest.raises(ValueError):
        associate(np.ones((2, 1, 4, 4)), [30, 0], [0, 0, 1, 1])


def test_tie_keeps_previous_index():
    hard = np.zeros((2, 4, 4), int)
    hard[0, :, :2] = 1
    hard[1] = 0
    hard[1, 2:, :] = 1
    masks = np.eye(3)[hard].transpose(0, 3, 1, 2)
    assert associate(masks, [0, 30], [0, 0, 0.5, 1]) == [1, 1]
