import pytest

from scripts.prepare_grid_development_split import allocate


def test_assignment_is_deterministic_group_disjoint_and_order_independent():
    speakers = [f's{i}' for i in range(2, 34)]
    split = allocate(speakers)
    assert split == allocate(list(reversed(speakers)))
    assert len(split['train']) == 24 and len(split['validation']) == 8
    assert not set(split['train']) & set(split['validation'])


def test_duplicate_speaker_fails_closed():
    with pytest.raises(ValueError, match='distinct'):
        allocate(['s1'] * 32)
