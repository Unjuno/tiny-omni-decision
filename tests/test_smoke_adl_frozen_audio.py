import sys

import numpy as np
import pytest

from scripts.audit_adl_participant_split import participant_split
from scripts.smoke_adl_frozen_audio import first_train_row, main


def test_selects_first_train_row_without_using_validation_or_unassigned():
    train = next(i for i in range(100) if participant_split(i) == 'train')
    valid = next(i for i in range(100) if participant_split(i) == 'validation')
    assert first_train_row(np.array([999, valid, train, train])) == 2


def test_no_train_row_fails():
    with pytest.raises(StopIteration):
        first_train_row(np.array([999]))


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    original = tmp_path / 'run.json'
    original.write_text('original', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['smoke', '--output', str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()
    assert original.read_text(encoding='utf-8') == 'original'
