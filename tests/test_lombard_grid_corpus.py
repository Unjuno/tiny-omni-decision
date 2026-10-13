import pytest

from scripts.lombard_grid_corpus import (
    decode_code,
    parse_member,
    partition_groups,
    reconcile_label,
    speaker_groups,
)


def test_native_slots_are_from_actual_speech_not_wrong_intended_prompt():
    row = parse_member('lombardgrid/audio/s13_l_bbbm6s_WRONG_bbbn6s.wav')
    assert row['actual_code'] == 'bbbn6s'
    assert row['intended_code'] == 'bbbm6s'
    assert row['targets']['letter'] == 'n'
    assert row['targets']['color'] == 'blue'


def test_unknown_annotation_remains_visible_without_invented_target():
    row = parse_member('lombardgrid/audio/s13_l_lbbr2n_WRONG_#(error).wav')
    assert row['targets'] is None
    assert row['actual_annotation'] == '#(error)'
    assert row['speaker'] == 's13'


@pytest.mark.parametrize('code', ['bbat9p', 'pgag6a', 'lrbizp'])
def test_code_decodes_all_six_native_slots(code):
    assert set(decode_code(code)) == {'command', 'color', 'preposition', 'letter',
                                      'number', 'adverb'}


@pytest.mark.parametrize('code', ['lgwo9az', 'bah2s', 'mistake', '#brbb5s(distorted)',
                                 'bbaw9p', 'bbat0p'])
def test_irregular_or_marked_transcription_is_not_silently_cleaned(code):
    with pytest.raises(ValueError):
        decode_code(code)


def test_explicit_repeated_recording_remains_a_distinct_asset():
    regular = parse_member('lombardgrid/audio/s14_p_prwr3a.wav')
    repeated = parse_member('lombardgrid/audio/s14_1_50_15_r_prwr3a.wav')
    assert regular['source_id'] != repeated['source_id']
    assert regular['actual_code'] == repeated['actual_code']
    assert regular['speaker'] == repeated['speaker']
    assert repeated['condition'] == 'p'


def test_speaker_and_repeated_actual_or_intended_sentence_stay_together():
    rows = [parse_member(name) for name in [
        's2_p_bbat9p.wav', 's3_l_bbat9p.wav',
        's3_p_pgag6a.wav', 's4_l_pgag6a_WRONG_lrbizp.wav',
        's5_p_lrbizp.wav', 's6_l_bbby1s.wav', 's29_p_praw4p.wav',
    ]]
    groups = speaker_groups(rows)
    assert groups['s2'] == groups['s3'] == groups['s4'] == groups['s5']
    assert groups['s6'] == groups['s29']
    assert groups == speaker_groups(list(reversed(rows)))


def test_unknown_member_structure_fails_instead_of_guessing():
    with pytest.raises(ValueError):
        parse_member('s14_2_52_15_r_prwr3a.wav')


def test_metadata_disagreement_is_quarantined_without_overriding_source():
    original = parse_member('s13_l_bwwx8a_WRONG_lbif7p.wav')
    result = reconcile_label(original, [{'STATUS': 'WRONG', 'TRANSCRIPTION': 'bwwx8a'}])
    assert result['targets'] is None
    assert result['label_status'] == 'metadata_conflict'
    assert original['targets'] is not None
    assert result['actual_code'] == original['actual_code']


def test_missing_metadata_is_explicit_and_unknown_transcription_not_guessed():
    result = reconcile_label(parse_member('s31_l_bbwg8a.wav'), [])
    assert result['label_origin'] == 'official_filename_only_metadata_missing'
    assert result['targets'] is not None
    unknown = reconcile_label(parse_member('s13_l_lbbr2n_WRONG_#(error).wav'), [])
    assert unknown['targets'] is None
    assert unknown['label_status'] == 'unresolved_transcription'


def test_group_partitions_are_deterministic_and_never_split_connected_speakers():
    rows = [parse_member(f's{i}_p_bbat9p.wav') for i in [2, 3, 4]]
    first = partition_groups(rows)
    assert first == partition_groups(list(reversed(rows)))
    assert len(set(first.values())) == 1
