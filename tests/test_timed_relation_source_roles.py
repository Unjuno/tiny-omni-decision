import copy

import pytest

from scripts import diagnose_timed_relations as probe


def source():
    return dict(grounded_question=[
        dict(question='Task. Track the occluded object.', answers=[7]),
        dict(question='Task. Track the object that occludes the smaller object.', answers=[3]),
        dict(question='Task. Track these similar or identical objects.', answers=[1, 2, 3])])


def test_source_roles_use_grounded_tracks_without_option_namespace():
    assert probe.grounded_roles(source()) == dict(hidden=[7], cover=[3], group=[1, 2, 3])


@pytest.mark.parametrize('answers', [[7], [3, 2], [True], ['3']])
def test_invalid_cover_namespace_or_cardinality_rejected(answers):
    s = source()
    s['grounded_question'][1]['answers'] = answers
    with pytest.raises(ValueError):
        probe.grounded_roles(s)


def test_duplicate_role_questions_not_silently_selected():
    s = source()
    s['grounded_question'].append(copy.deepcopy(s['grounded_question'][0]))
    with pytest.raises(ValueError):
        probe.grounded_roles(s)


def test_variable_group_sizes_preserved_and_hidden_role_distinct():
    s = source()
    s['grounded_question'][2]['answers'] = [1, 2, 3, 4]
    assert probe.grounded_roles(s)['group'] == [1, 2, 3, 4]
    s['grounded_question'][0]['answers'] = [4]
    with pytest.raises(ValueError):
        probe.grounded_roles(s)
