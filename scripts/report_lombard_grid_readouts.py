"""Fixed-model, whole-speaker-group comparison helpers; no optimization on controls."""

import numpy as np

from scripts.lombard_readout_core import (
    OPTIONS,
    _log_probabilities,
    selected_epoch,
    summarize,
    targets_from_rows,
)
from scripts.starss23_readout_common import different_recording_permutation


def _identities(rows):
    ids = [r['source_id'] for r in rows]
    if (not rows or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids)
            or any(not isinstance(r['speaker_group'], str)
                   or not r['speaker_group'] for r in rows)):
        raise ValueError('unique source IDs and nonempty whole-speaker-group identities required')


def source_permutation(rows):
    """Preserve plain/Lombard condition as well as the exact donor marginal."""
    _identities(rows)
    if any(r['condition'] not in {'p', 'l'} for r in rows):
        raise ValueError('native p/l conditions required for source control')
    result = list(range(len(rows)))
    for condition in ['p', 'l']:
        indices = [i for i, r in enumerate(rows) if r['condition'] == condition]
        if not indices:
            continue
        local = different_recording_permutation([rows[i]['speaker_group'] for i in indices], 17)
        for original, replacement in zip(indices, local, strict=True):
            result[original] = indices[replacement]
    if sorted(result) != list(range(len(rows))) or any(
            rows[i]['speaker_group'] == rows[j]['speaker_group'] for i, j in enumerate(result)):
        raise ValueError('cross-speaker bijection invariant failed')
    return result


def _group_statistics(logits, targets, rows, groups, temperatures):
    logp = _log_probabilities(logits, temperatures)
    if len(logp) != len(rows):
        raise ValueError('paired logit/source row alignment differs')
    p = np.exp(logp)
    n, correct, losses, briers, wholes, bins = [], [], [], [], [], []
    boundaries = np.cumsum([0, *[len(v) for v in OPTIONS.values()]])
    for group in groups:
        index = np.array([i for i, r in enumerate(rows) if r['speaker_group'] == group])
        n.append(len(index))
        slot_correct, slot_losses, slot_brier, slot_bins, complete = [], [], [], [], []
        for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            values = p[index, start:end]
            y = targets[index, j]
            good, conf = values.argmax(1) == y, values.max(1)
            complete.append(good)
            slot_correct.append(good.sum())
            slot_losses.append(-logp[index, start+y].sum())
            brier = values.copy()
            brier[np.arange(len(index)), y] -= 1
            slot_brier.append(np.square(brier).sum())
            membership = np.minimum((conf*15).astype(int), 14)
            # ECE's count*(accuracy-confidence) reduces to correct_sum-confidence_sum.
            slot_bins.append([float((good[membership == b]-conf[membership == b]).sum())
                              for b in range(15)])
        correct.append(slot_correct)
        losses.append(slot_losses)
        briers.append(slot_brier)
        bins.append(slot_bins)
        wholes.append(np.stack(complete, 1).all(1).sum())
    return [np.asarray(v, dtype=np.float64) for v in [n, correct, losses, briers, wholes, bins]]


def _aggregate(stats, weights):
    n, correct, nll, brier, whole, bins = stats
    denominator = weights@n
    accuracy = (weights@correct)/denominator[:, None]
    ece = abs(weights@bins.reshape(len(n), -1)).reshape(len(weights), 6, 15).sum(2)
    return dict(macro_accuracy=accuracy.mean(1), minimum_slot_accuracy=accuracy.min(1),
                macro_nll=((weights@nll)/denominator[:, None]).mean(1),
                macro_brier=((weights@brier)/denominator[:, None]).mean(1),
                macro_ece=(ece/denominator[:, None]).mean(1),
                whole_utterance_accuracy=(weights@whole)/denominator)


def paired_bootstrap(first, second, rows, first_temperatures=None, second_temperatures=None,
                     *, replicates=10000, seed=17):
    _identities(rows)
    if (not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1):
        raise ValueError('positive bootstrap replicate count required')
    y = targets_from_rows(rows)
    groups = sorted({r['speaker_group'] for r in rows})
    stats = [_group_statistics(v, y, rows, groups, t) for v, t in [
        (first, first_temperatures), (second, second_temperatures)]]
    # Resample complete groups, retaining all their observations; no row-level bootstrap.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
    weights = np.stack([(draws == j).sum(1) for j in range(len(groups))], axis=1)
    resampled = [_aggregate(s, weights) for s in stats]
    point = [_aggregate(s, np.ones((1, len(groups)))) for s in stats]
    differences = {}
    for key in point[0]:
        delta = resampled[1][key]-resampled[0][key]
        lower, upper = np.quantile(delta, [.025, .975], method='linear')
        differences[key] = dict(point=float(point[1][key][0]-point[0][key][0]),
                                lower=float(lower), upper=float(upper))
    return dict(groups=len(groups), group_ids=groups, observations=len(rows), seed=seed,
                replicates=replicates, method='paired whole-group percentile95%, linear quantiles',
                difference_second_minus_first=differences,
                scope='development validation and selected checkpoints; not blind audit')


def validate_saved_predictions(logits, records, rows, temperatures=None):
    _identities(rows)
    if [r['source_id'] for r in records] != [r['source_id'] for r in rows]:
        raise ValueError('saved prediction IDs/order differ')
    metric, probabilities = summarize(logits, targets_from_rows(rows), temperatures)
    start = 0
    for slot, options in OPTIONS.items():
        end = start+len(options)
        for i, (record, source) in enumerate(zip(records, rows, strict=True)):
            if any(record[k] != source[k]
                   for k in ['speaker', 'speaker_group', 'condition', 'targets']):
                raise ValueError('saved prediction source/target binding differs')
            if (set(record['probabilities']) != set(OPTIONS)
                    or set(record['predictions']) != set(OPTIONS)
                    or list(record['probabilities'][slot]) != options):
                raise ValueError('all native options/predictions in canonical order required')
            expected = probabilities[i, start:end]
            saved = np.asarray(list(record['probabilities'][slot].values()))
            if (not np.array_equal(saved, expected)
                    or record['predictions'][slot] != options[int(expected.argmax())]):
                raise ValueError('stored probability/prediction differs from full logits')
        start = end
    return metric


def _audit_metric(value, count):
    if set(value['by_slot']) != set(OPTIONS) or value['macro']['count'] != count*6:
        raise ValueError('full native metric counts/slots differ')
    keys = ['accuracy', 'nll', 'brier', 'ece', 'mean_confidence']
    for slot in OPTIONS:
        m = value['by_slot'][slot]
        if m['count'] != count or not np.isfinite([m[k] for k in keys]).all():
            raise ValueError('invalid per-slot metric count/value')
        if (not 0 <= m['accuracy'] <= 1 or not 0 <= m['ece'] <= 1
                or not 0 <= m['mean_confidence'] <= 1 or not 0 <= m['brier'] <= 2
                or m['nll'] < 0):
            raise ValueError('per-slot metric outside valid range')
    for key in keys:
        expected = np.mean([m[key] for m in value['by_slot'].values()])
        if not np.isclose(value['macro'][key], expected, atol=1e-12, rtol=0):
            raise ValueError('macro metrics differ from six equal native slots')
    minimum = min(m['accuracy'] for m in value['by_slot'].values())
    if (value['minimum_slot_accuracy'] != minimum
            or not 0 <= value['whole_utterance_accuracy'] <= minimum):
        raise ValueError('minimum/whole-utterance metric inconsistent')


def audit_summary(report, provenance, *, train_count=4080, validation_count=889):
    from scripts.train_lombard_grid_readouts import accounting
    if (report.get('status') != 'PASS_ARM_EXECUTION' or report.get('smoke') is not False
            or report['provenance'] != provenance or report['protocol'] != provenance['protocol']
            or provenance['training_only_smoke'] is not False
            or report['protocol']['epochs'] != 24 or report['protocol']['batch_size'] != 128
            or report['head_parameters'] != 77235
            or report['encoders_loaded_in_training'] is not False
            or any(report.get(k) is not True for k in ['validation_evaluated',
                    'calibration_evaluated', 'reload_bitwise_equal',
                    'json_option_permutation_exact'])):
        raise ValueError('complete fixed-budget calibrated/reloaded full arm required')
    history = report['history']
    if [r['epoch'] for r in history] != list(range(1, 25)):
        raise ValueError('all24 training/validation observations required')
    for row in history:
        epoch = row['epoch']
        if (row['consumed_assets'] != epoch*train_count
                or row['updates'] != accounting(train_count, epoch)['optimizer_updates']):
            raise ValueError('history consumption/update count differs')
        _audit_metric(row['train_fixed'], train_count)
        _audit_metric(row['validation'], validation_count)
    best = selected_epoch(history)
    if (report['best_epoch'] != best
            or report['accounting'] != accounting(train_count, 24, best)):
        raise ValueError('selected checkpoint or total/selected consumption differs')
    temperatures = np.asarray(report['temperatures'])
    if (temperatures.shape != (6,) or not np.isfinite(temperatures).all()
            or (temperatures <= 0).any()):
        raise ValueError('six fixed-calibration temperatures required')


def audit_optimizer_history(entries, *, train_count=4080):
    batches = (train_count+127)//128
    if len(entries) != 24*batches or [e['update'] for e in entries] != list(range(1, 24*batches+1)):
        raise ValueError('complete ordered optimizer log required')
    for index, entry in enumerate(entries):
        epoch, batch = index//batches+1, index % batches
        count = min(128, train_count-batch*128)
        if (entry['epoch'] != epoch or entry['samples'] != count
                or entry['learning_rate'] != .001
                or not np.isfinite(entry['gradient_norm']) or entry['gradient_norm'] < 0):
            raise ValueError('actual sample/LR/gradient update differs from fixed budget')
