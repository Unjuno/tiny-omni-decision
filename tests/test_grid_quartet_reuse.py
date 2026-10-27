import pytest

from scripts.audit_grid_quartet_reuse import summarize


def test_counts_audio_reuse_and_speaker_groups():
    row = {'speaker': 's1', 'codes': ['a', 'b', 'c', 'd'],
           'queries': [{'audio_code': c} for c in 'abcd']}
    result = summarize([row, row])
    assert result['quartets'] == 2 and result['speaker_labels'] == 1
    assert result['unique_audio_codes'] == 4 and result['max_roles_per_audio'] == 2


def test_invalid_queries_fail_closed():
    with pytest.raises(ValueError, match='query roles'):
        summarize([{'speaker': 's1', 'codes': ['a'], 'queries': []}])
