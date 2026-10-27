import pytest

from scripts.audit_grid_lombard_index import compare


def test_counts_distinct_codes_and_rows():
    result = compare(['s1/align/abcdef', 's2/align/abcdef'],
                     [{'actual_code': 'abcdef'}, {'actual_code': 'abcdef'}])
    assert result['overlap_unique_codes'] == 1
    assert result['overlap_train_rows'] == 2


def test_reject_bad_index():
    with pytest.raises(ValueError, match='invalid'):
        compare(['s1/align/abcdef', 's1/align/abcdef'], [])
