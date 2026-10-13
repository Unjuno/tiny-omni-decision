"""Fixed-model, whole-speaker-group comparison helpers; no optimization on controls."""

import numpy as np

from scripts.lombard_readout_core import (
    OPTIONS,
    _log_probabilities,
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
