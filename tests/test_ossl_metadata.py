import pytest

from scripts.audit_ossl_metadata import summarize


def test_source_overlap_and_train_duplicates():
    rows = [
        {'movie_id': '1', 'split': 'train', 'youtube_id': 'a', 'n_clips': '3'},
        {'movie_id': '2', 'split': 'train', 'youtube_id': 'a', 'n_clips': '2'},
        {'movie_id': '3', 'split': 'test', 'youtube_id': 'a', 'n_clips': '1'},
    ]
    result = summarize(rows)
    assert result['train_clips'] == 5
    assert result['train_duplicate_source_ids'] == 1
    assert result['train_test_shared_source_ids'] == 1


def test_reject_duplicate_film_id():
    rows = [
        {'movie_id': '1', 'split': 'train'},
        {'movie_id': '1', 'split': 'test'},
    ]
    with pytest.raises(ValueError, match='duplicate'):
        summarize(rows)
