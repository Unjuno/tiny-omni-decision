import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.probe_droid_instruction import group_components, metrics, word_features


def test_transitive_instruction_groups():
    groups = group_components([['a'], ['a', 'b'], ['b'], [], []])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[4]


def test_word_hash_and_missing_indicator():
    a = word_features(['red cube', '', 'red cube'])
    assert np.array_equal(a[0], a[2])
    assert a[1, -1] == 1 and a[0, -1] == 0


def test_perfect_probability_metrics():
    p = np.eye(7)[[1, 2]]
    result = metrics(p, np.array([1, 2]))
    assert result['accuracy'] == 1 and result['nll'] == 0 and result['ece'] == 0
