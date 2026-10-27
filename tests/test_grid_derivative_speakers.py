import pytest

from scripts.audit_grid_derivative_speakers import ORIGINAL, compare, parse_unique


def test_code_join_finds_renamed_speaker():
    result = compare({'abcdef': 's1', 'abcde1': 's2'},
                     {'abcdef': 's3', 'abcde1': 's2'})
    assert result['shared_different_speaker_labels'] == 1
    assert result['different_speaker_labels'] == 1


def test_duplicate_code_rejected():
    names = ['alignments/s1/bbaf2n.align', 'alignments/s2/bbaf2n.align']
    with pytest.raises(ValueError, match='duplicate'):
        parse_unique(names, ORIGINAL)
