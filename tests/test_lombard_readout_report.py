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
    audit_optimizer_history,
    audit_summary,
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


@pytest.mark.parametrize('fault', ['none', 'smoke', 'budget', 'selected', 'accounting',
                                 'macro', 'calibration', 'protocol', 'reload'])
def test_full_run_contract_does_not_accept_partial_or_selected_count_confusion(fault):
    from scripts.train_lombard_grid_readouts import accounting
    def metric(count, nll):
        slots = {s: dict(count=count, accuracy=.5, nll=nll, brier=.8,
                         ece=.1, mean_confidence=.6) for s in OPTIONS}
        return dict(by_slot=slots, macro={**slots['command'], 'count': count*6},
                    minimum_slot_accuracy=.5, whole_utterance_accuracy=.1)
    protocol = dict(epochs=24, batch_size=128)
    provenance = dict(protocol=protocol, training_only_smoke=False)
    history = [dict(epoch=e, consumed_assets=e*4, updates=e,
                    train_fixed=metric(4, .1), validation=metric(6, .2 if e == 7 else .3))
               for e in range(1, 25)]
    report = dict(status='PASS_ARM_EXECUTION', smoke=False, protocol=protocol,
                  provenance=provenance, head_parameters=77235,
                  encoders_loaded_in_training=False, validation_evaluated=True,
                  calibration_evaluated=True, reload_bitwise_equal=True,
                  json_option_permutation_exact=True, history=history, best_epoch=7,
                  accounting=accounting(4, 24, 7), temperatures=[1.]*6)
    if fault == 'smoke':
        report['smoke'] = True
    elif fault == 'budget':
        report['history'].pop()
    elif fault == 'selected':
        report['best_epoch'] = 24
    elif fault == 'accounting':
        report['accounting']['consumed_assets'] = 28
    elif fault == 'macro':
        report['history'][0]['validation']['macro']['nll'] = 0
    elif fault == 'calibration':
        report['calibration_evaluated'] = False
    elif fault == 'protocol':
        report['protocol'] = {'epochs': 512}
    elif fault == 'reload':
        report['reload_bitwise_equal'] = False
    if fault == 'none':
        audit_summary(report, provenance, train_count=4, validation_count=6)
    else:
        with pytest.raises(ValueError):
            audit_summary(report, provenance, train_count=4, validation_count=6)


@pytest.mark.parametrize('fault', ['none', 'missing', 'samples', 'lr', 'gradient', 'order'])
def test_optimizer_log_matches_actual_full_budget_and_finite_updates(fault):
    entries = [dict(update=i, epoch=i, samples=4, gradient_norm=1., learning_rate=.001)
               for i in range(1, 25)]
    if fault == 'missing':
        entries.pop()
    elif fault == 'samples':
        entries[0]['samples'] = 3
    elif fault == 'lr':
        entries[0]['learning_rate'] = .01
    elif fault == 'gradient':
        entries[0]['gradient_norm'] = float('nan')
    elif fault == 'order':
        entries.reverse()
    if fault == 'none':
        audit_optimizer_history(entries, train_count=4)
    else:
        with pytest.raises(ValueError):
            audit_optimizer_history(entries, train_count=4)
