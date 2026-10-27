import numpy as np
import pytest

from scripts.audit_adl_participant_split import participant_split
from scripts.smoke_adl_frozen_audio import first_train_row


def test_selects_first_train_row_without_using_validation_or_unassigned():
    train = next(i for i in range(100) if participant_split(i) == 'train')
    valid = next(i for i in range(100) if participant_split(i) == 'validation')
    assert first_train_row(np.array([999, valid, train, train])) == 2


def test_no_train_row_fails():
    with pytest.raises(StopIteration):
        first_train_row(np.array([999]))
