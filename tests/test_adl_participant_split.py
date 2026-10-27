import pytest

from scripts.audit_adl_participant_split import (
    participant_split,
    protocol_digest,
    summarize_activity,
)


def test_fixed_hash_assignment_and_no_participant_handling():
    assert participant_split(11) == 'validation'
    assert participant_split(2) == 'train'
    assert participant_split(999) is None
    assert participant_split(11) == participant_split(11)


@pytest.mark.parametrize('participant', [-1, 1.5, True, '1'])
def test_invalid_participant_ids_fail_closed(participant):
    with pytest.raises(ValueError):
        participant_split(participant)


def test_activity_counts_and_unassigned_rows_are_preserved():
    result = summarize_activity([11, 11, 2, 999], [1, 2, 1, 1])
    assert result['validation'] == {'participants': 1, 'events': 2}
    assert result['train'] == {'participants': 1, 'events': 1}
    assert result['unassigned_events'] == 1
    assert result['coverage_pass'] is True


def test_duplicate_identity_or_missing_class_group_fails_coverage():
    with pytest.raises(ValueError):
        summarize_activity([1, 1], [5, 5])
    assert summarize_activity([1], [1])['coverage_pass'] is False


def test_protocol_hash_is_stable_across_checkout_line_endings(tmp_path):
    path = tmp_path / 'protocol.md'
    path.write_bytes(b'first\nsecond\n')
    expected = protocol_digest(path)
    path.write_bytes(b'first\r\nsecond\r\n')
    assert protocol_digest(path) == expected
    path.write_bytes(b'first\nchanged\n')
    assert protocol_digest(path) != expected
