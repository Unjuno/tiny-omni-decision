import copy
import json

import numpy as np
import pytest

pytest.importorskip('torch')

from scripts.lombard_readout_core import (  # noqa: E402
    OPTIONS,
    json_decision,
    summarize,
    targets_from_rows,
)
from scripts.report_lombard_grid_readouts import (  # noqa: E402
    paired_bootstrap,
    source_permutation,
    validate_saved_predictions,
)


def rows():
    result = []
    for i, group in enumerate(['a', 'a', 'b', 'b', 'c', 'c']):
        result.append(dict(source_id=str(i), speaker_group=group, speaker=group,
                           condition='p' if i % 2 == 0 else 'l',
                           targets={s: o[-1] for s, o in OPTIONS.items()}))
    return result


def test_mismatch_is_label_independent_bijective_cross_group_and_condition_matched():
    records = rows()
    permutation = source_permutation(records)
    assert sorted(permutation) == list(range(len(records)))
    for i, j in enumerate(permutation):
        assert records[i]['speaker_group'] != records[j]['speaker_group']
        assert records[i]['condition'] == records[j]['condition']
    for row in records:
        row['targets'] = 'unused corrupted labels'
    assert permutation == source_permutation(records)


def test_equal_models_have_exact_zero_paired_intervals_and_seed_repeatability():
    values = np.zeros((6, 51))
    first = paired_bootstrap(values, values, rows(), replicates=200)
    second = paired_bootstrap(values, values, rows(), replicates=200)
    assert first == second
    assert first['groups'] == 3 and first['seed'] == 17 and first['replicates'] == 200
    for metric in first['difference_second_minus_first'].values():
        assert metric == {'point': 0., 'lower': 0., 'upper': 0.}


def test_group_bootstrap_keeps_all_six_native_tasks_and_reports_direction():
    first, second = np.zeros((6, 51)), np.zeros((6, 51))
    start = 0
    for options in OPTIONS.values():
        second[:, start + len(options)-1] = 10
        start += len(options)
    result = paired_bootstrap(first, second, rows(), replicates=200)
    delta = result['difference_second_minus_first']
    assert delta['macro_accuracy']['point'] == 1.
    assert delta['whole_utterance_accuracy']['point'] == 1.
    assert delta['macro_nll']['upper'] < 0 and delta['macro_brier']['upper'] < 0


def test_duplicate_source_or_impossible_derangement_is_rejected():
    records = rows()
    records[1]['source_id'] = records[0]['source_id']
    with pytest.raises(ValueError):
        source_permutation(records)
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros((6, 51)), np.zeros((6, 51)), records, replicates=10)
    records = rows()
    for row in records:
        row['speaker_group'] = 'same'
    with pytest.raises(ValueError):
        source_permutation(records)


def test_whole_group_point_metrics_equal_direct_unequal_group_metrics():
    records = rows()[:5]
    values = np.random.default_rng(17).normal(size=(5, 51))
    baseline = np.zeros_like(values)
    result = paired_bootstrap(baseline, values, records, replicates=200)
    a = summarize(baseline, targets_from_rows(records))[0]
    b = summarize(values, targets_from_rows(records))[0]
    for key in ['accuracy', 'nll', 'brier', 'ece']:
        assert result['difference_second_minus_first']['macro_'+key]['point'] == pytest.approx(
            b['macro'][key]-a['macro'][key], abs=1e-12)


@pytest.mark.parametrize('fault', ['none', 'order', 'target', 'probability',
                                 'prediction', 'options'])
def test_saved_prediction_audit_recomputes_full_logits_and_rejects_corruption(fault):
    source = rows()
    values = np.arange(6*51, dtype=float).reshape(6, 51)
    records = []
    for i, row in enumerate(source):
        p = json.loads(json_decision(values[i]))
        records.append({**copy.deepcopy(row), 'probabilities': p,
                        'predictions': {s: max(v, key=v.__getitem__) for s, v in p.items()}})
    if fault == 'order':
        records.reverse()
    elif fault == 'target':
        records[0]['targets']['command'] = 'bin'
    elif fault == 'probability':
        records[0]['probabilities']['letter']['a'] += .1
    elif fault == 'prediction':
        records[0]['predictions']['letter'] = 'a'
    elif fault == 'options':
        del records[0]['probabilities']['letter']['z']
    if fault == 'none':
        metric = validate_saved_predictions(values, records, source)
        assert metric == summarize(values, targets_from_rows(source))[0]
    else:
        with pytest.raises(ValueError):
            validate_saved_predictions(values, records, source)
