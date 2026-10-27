import pytest

from scripts.audit_adl_partial_inventory import select_jobs, summarize_inventory


def make_entries():
    return [{'key': f'{i:02d}_activity.zip', 'size': 70000 + i,
             'checksum': 'md5:' + 'a' * 32,
             'links': {'self': f'https://example.org/{i}'}} for i in range(25)]


def test_selects_every_numbered_activity_once_in_fixed_size_order():
    entries = make_entries()
    entries[3]['size'] = 66000
    jobs = select_jobs(entries + [{'key': 'README.md'}])
    assert len(jobs) == 25
    assert jobs[0]['key'] == '03_activity.zip'
    assert {job['key'][:2] for job in jobs} == {f'{i:02d}' for i in range(25)}


def test_missing_duplicate_or_insecure_sources_fail_closed():
    entries = make_entries()
    with pytest.raises(ValueError):
        select_jobs(entries[:-1])
    with pytest.raises(ValueError):
        select_jobs(entries + [entries[0]])
    entries[0]['links']['self'] = 'http://example.org/unsafe'
    with pytest.raises(ValueError):
        select_jobs(entries)


def test_unassigned_and_unsupported_never_count_as_complete():
    summaries = [{'status': 'PASS_MEMBER_CRC', 'split': {'coverage_pass': True,
                                                          'unassigned_events': 0}},
                 {'status': 'PASS_MEMBER_CRC', 'split': {'coverage_pass': False,
                                                          'unassigned_events': 50}},
                 {'status': 'UNSUPPORTED'}]
    result = summarize_inventory(summaries, expected=3)
    assert result['complete'] is False
    assert result['unsupported_or_failed'] == 1
    assert result['classes_without_both_splits'] == 1
    assert result['unassigned_events'] == 50
