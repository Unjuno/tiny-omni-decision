import pytest

from scripts.audit_grid_audio_folder_mapping import audit_mapping, parse_audio_members


def test_audio_member_parser_ignores_macos_sidecars_and_directories():
    members = [
        'audio_25k/',
        'audio_25k/s1/',
        'audio_25k/s1/ab123c.wav',
        '__MACOSX/audio_25k/s1/._ab123c.wav',
        'audio_25k/s2/xy987z.wav',
    ]
    assert parse_audio_members(members) == {
        's1': {'ab123c'},
        's2': {'xy987z'},
    }


def test_audit_mapping_detects_cross_folder_permutation_without_repairing_it():
    rows = [
        {'speaker': 's1', 'code': 'ab123c'},
        {'speaker': 's1', 'code': 'de456f'},
        {'speaker': 's2', 'code': 'xy987z'},
    ]
    audio = {'s1': {'xy987z'}, 's2': {'ab123c', 'de456f'}}
    result = audit_mapping(rows, audio)
    assert result['status'] == 'BLOCKED_AUDIO_SPEAKER_FOLDER_MAPPING'
    assert result['self_folder_missing_codes'] == 3
    assert result['folder_matches']['s1']['best_audio_folder'] == 's2'
    assert result['folder_matches']['s1']['best_overlap'] == 2
    assert result['folder_matches']['s2']['best_audio_folder'] == 's1'


def test_audit_mapping_passes_identity_mapping_and_rejects_duplicate_source_rows():
    rows = [{'speaker': 's1', 'code': 'ab123c'}]
    assert audit_mapping(rows, {'s1': {'ab123c'}})['status'] == 'PASS_AUDIO_FOLDER_MAPPING'
    with pytest.raises(ValueError, match='duplicate'):
        audit_mapping(rows * 2, {'s1': {'ab123c'}})
