import copy

import numpy as np
import pytest

from scripts.clap_projection import fit_train_projection, project, validate_coverage


def test_projection_fits_only_unique_train_assets_and_is_repeatable():
    values = np.random.default_rng(17).normal(size=(9, 5))
    ids = [str(i) for i in range(9)]
    train = ids[:7]
    first = fit_train_projection(values, ids, train, 3)
    changed = values.copy()
    changed[7:] = 1e6
    second = fit_train_projection(changed, ids, train, 3)
    for key in ('mean', 'components', 'singular_values'):
        np.testing.assert_array_equal(first[key], second[key])
    assert first['fit_ids'] == train
    assert first['components'].shape == (3, 5)
    np.testing.assert_allclose(first['components'] @ first['components'].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(project(values[:7], first).mean(0), 0, atol=1e-7)
    assert project(values, first).dtype == np.float32


@pytest.mark.parametrize('train', [['0', '0', '1'], ['0', 'unknown', '2'], ['0']])
def test_bad_train_identity_or_insufficient_capacity_fails_closed(train):
    with pytest.raises(ValueError):
        fit_train_projection(np.eye(4), ['0', '1', '2', '3'], train, 2)


def fixture():
    records = [{'source_id': 'x', 'split': 'train'}]
    report = {'status': 'PASS', 'scope': 'full', 'repeat_bitwise_equal': True,
              'rows': [{'id': 'x', 'split': 'train', 'samples': 11,
                        'covered_samples': 11, 'spans': [[0, 10], [10, 11]],
                        'window_offset': 0, 'windows': 2}], 'window_samples': 10}
    windows = np.eye(2, dtype=np.float32)
    pooled = np.array([[10, 1]], dtype=np.float32) / np.sqrt(101)
    return records, report, pooled, windows


def test_complete_coverage_accepts_tail_and_rechecks_pooling():
    validate_coverage(*fixture())


@pytest.mark.parametrize('change', ['gap', 'wrong_split', 'offset', 'coverage', 'nonfinite',
                                   'wrong_pool', 'extra_window', 'incomplete'])
def test_coverage_rejects_corrupt_or_partial_extraction(change):
    records, report, pooled, windows = copy.deepcopy(fixture())
    row = report['rows'][0]
    if change == 'gap':
        row['spans'][1][0] = 9
    elif change == 'wrong_split':
        row['split'] = 'validation'
    elif change == 'offset':
        row['window_offset'] = 1
    elif change == 'coverage':
        row['covered_samples'] = 10
    elif change == 'nonfinite':
        windows[0, 0] = np.nan
    elif change == 'wrong_pool':
        pooled[:] = 1
    elif change == 'extra_window':
        windows = np.concatenate([windows, windows[:1]])
    else:
        report['status'] = 'RUNNING'
    with pytest.raises(ValueError):
        validate_coverage(records, report, pooled, windows)
