from scripts.audit_grid_alignment_content import words_match


def test_alignment_words_match_normalized_numeric_target():
    target = {'command': 'bin', 'color': 'blue', 'preposition': 'at',
              'letter': 'f', 'number': '2', 'adverb': 'now'}
    assert words_match(['bin', 'blue', 'at', 'f', 'two', 'now'], target)
    assert not words_match(['bin', 'blue', 'at', 'f', 'three', 'now'], target)
