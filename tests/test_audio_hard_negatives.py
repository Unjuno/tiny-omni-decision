from tiny_omni_decision.audio_hard_negatives import (
    TranscriptRecord,
    find_hard_distractors,
    make_transcript_record,
    normalize_transcript,
    parse_cmudict,
    transcript_phonemes,
)

CMUDICT_SAMPLE = """;;; sample dictionary
A  AH0
CAT  K AE1 T
DOG  D AO1 G
FOG  F AO1 G
HAT  HH AE1 T
LOG  L AO1 G
RED  R EH1 D
SEES  S IY1 Z
THE  DH AH0
THE(2)  DH IY0
"""


def _record(sample_id: str, speaker: str, chapter: str, text: str, dictionary):
    record = make_transcript_record(sample_id, speaker, chapter, text, dictionary)
    assert record is not None
    return record


def test_cmudict_parser_normalizes_stress_and_alternate_pronunciation() -> None:
    dictionary = parse_cmudict(CMUDICT_SAMPLE)

    assert dictionary["CAT"] == ("K", "AE", "T")
    assert dictionary["THE"] == ("DH", "AH")
    assert normalize_transcript("The cat, sees a red dog!") == "THE CAT SEES A RED DOG"
    assert transcript_phonemes("the cat", dictionary) == (
        ("THE", "CAT"),
        ("DH", "AH", "|", "K", "AE", "T"),
    )


def test_transcript_phonemes_fails_closed_for_unknown_words() -> None:
    assert transcript_phonemes("THE UNKNOWN CAT", parse_cmudict(CMUDICT_SAMPLE)) is None


def test_hard_distractors_match_phoneme_vocabulary_length_and_source_isolation() -> None:
    dictionary = parse_cmudict(CMUDICT_SAMPLE)
    target = _record("1-1-1", "1", "1", "THE CAT SEES A RED DOG", dictionary)
    candidates = [
        _record("2-2-1", "2", "2", "THE CAT SEES A RED FOG", dictionary),
        _record("3-3-1", "3", "3", "THE HAT SEES A RED DOG", dictionary),
        _record("4-4-1", "4", "4", "THE CAT SEES A RED LOG", dictionary),
        _record("5-5-1", "5", "5", "THE CAT SEES A DOG RED", dictionary),
        _record("1-9-1", "1", "9", "THE CAT SEES A RED FOG", dictionary),
        _record("8-8-1", "8", "8", "THE CAT", dictionary),
        _record("9-9-1", "9", "9", "THE CAT SEES A RED DOG", dictionary),
    ]

    matches = find_hard_distractors(target, candidates, count=3)

    assert [match.sample_id for match in matches] == ["2-2-1", "3-3-1", "4-4-1"]
    assert all(match.phoneme_similarity > 0.7 for match in matches)
    assert all(match.vocabulary_jaccard > 0.5 for match in matches)
    assert all(match.word_count_delta == 0 for match in matches)
    assert all(match.character_length_ratio >= 0.65 for match in matches)
    assert all(match.sample_id != "1-9-1" for match in matches)
    assert all(match.sample_id != "9-9-1" for match in matches)


def test_hard_distractor_count_must_be_positive() -> None:
    record = TranscriptRecord("1", "s1", "c1", "A CAT", ("A", "CAT"), ("AH", "K", "AE", "T"))

    try:
        find_hard_distractors(record, [], count=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected positive distractor count validation")
