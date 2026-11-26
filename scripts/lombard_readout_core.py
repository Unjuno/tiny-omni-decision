"""Fixed native full-vocabulary slot readout; no filenames or text as features."""

import json

import numpy as np
import torch
from torch import nn

from scripts.lombard_grid_corpus import SLOTS

OPTIONS = {slot: list(table.values()) for slot, table in SLOTS.items()}
SIZES = tuple(len(options) for options in OPTIONS.values())
BOUNDARIES = np.cumsum((0, *SIZES))


class SlotHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(1154, 64), nn.GELU(), nn.Linear(64, 51))

    def forward(self, value):
        if value.ndim != 2 or value.shape[1] != 1154 or not torch.isfinite(value).all():
            raise ValueError('finite1154-dimensional observation features required')
        return super().forward(value)


def _feature(value, width):
    value = np.asarray(value)
    if (value.ndim != 2 or not len(value) or value.shape[1] != width
            or value.dtype != np.float32 or not np.isfinite(value).all()):
        raise ValueError('finite nonempty FP32 native features required')
    return value


def fit_normalizer(audio, video):
    audio, video = _feature(audio, 384), _feature(video, 768)
    if len(audio) != len(video):
        raise ValueError('aligned native feature rows required')
    result = {}
    for key, value in [('audio', audio), ('video', video)]:
        result[key + '_mean'] = value.mean(0, dtype=np.float64).astype(np.float32)
        std = value.std(0, dtype=np.float64)
        result[key + '_scale'] = np.where(std > 1e-6, std, 1).astype(np.float32)
    return result


def make_inputs(audio, video, condition, normalizer):
    if condition not in {'audio', 'video', 'joint'}:
        raise ValueError('unknown fixed comparison arm')
    audio, video = _feature(audio, 384), _feature(video, 768)
    if len(audio) != len(video):
        raise ValueError('aligned native feature rows required')
    result = []
    for key, value, present in [('audio', audio, condition != 'video'),
                                ('video', video, condition != 'audio')]:
        mean, scale = normalizer[key + '_mean'], normalizer[key + '_scale']
        if (mean.shape != (value.shape[1],) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError('invalid train normalizer')
        result.append((value - mean) / scale if present else np.zeros_like(value))
    bits = np.tile([condition != 'video', condition != 'audio'], (len(audio), 1))
    combined = np.concatenate([*result, bits.astype(np.float32)], axis=1).astype(np.float32)
    if not np.isfinite(combined).all():
        raise ValueError('nonfinite standardized features')
    return combined


def targets_from_rows(rows):
    values = []
    for row in rows:
        targets = row.get('targets')
        if not isinstance(targets, dict) or set(targets) != set(OPTIONS):
            raise ValueError('all six native slot targets required')
        if any(targets[s] not in opts for s, opts in OPTIONS.items()):
            raise ValueError('unknown native target; no vocabulary reduction')
        values.append([opts.index(targets[s]) for s, opts in OPTIONS.items()])
    if not values:
        raise ValueError('nonempty native target rows required')
    return np.asarray(values, dtype=np.int64)


def _log_probabilities(logits, temperatures=None):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not len(logits) or logits.shape[1] != 51:
        raise ValueError('nonempty51-logit batch required')
    if not np.isfinite(logits).all():
        raise ValueError('finite logits required')
    temps = np.ones(6) if temperatures is None else np.asarray(temperatures, dtype=np.float64)
    if temps.shape != (6,) or not np.isfinite(temps).all() or (temps <= 0).any():
        raise ValueError('six positive finite temperatures required')
    blocks = []
    for j, (start, end) in enumerate(zip(BOUNDARIES[:-1], BOUNDARIES[1:], strict=True)):
        value = logits[:, start:end] / temps[j]
        value = value - value.max(1, keepdims=True)
        blocks.append(value - np.log(np.exp(value).sum(1, keepdims=True)))
    result = np.concatenate(blocks, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite normalized logits')
    return result


def _targets(targets, count):
    y = np.asarray(targets)
    if (y.shape != (count, 6) or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= np.asarray(SIZES)).any()):
        raise ValueError('integer targets must index all six native vocabularies')
    return y


def summarize(logits, targets, temperatures=None):
    logp = _log_probabilities(logits, temperatures)
    y, p = _targets(targets, len(logp)), np.exp(logp)
    by_slot, correct_slots = {}, []
    for j, (slot, start, end) in enumerate(zip(OPTIONS, BOUNDARIES[:-1],
                                             BOUNDARIES[1:], strict=True)):
        value = p[:, start:end]
        predicted = value.argmax(1)
        correct, confidence = predicted == y[:, j], value.max(1)
        correct_slots.append(correct)
        brier = value.copy()
        brier[np.arange(len(y)), y[:, j]] -= 1
        bins = np.minimum((confidence * 15).astype(int), 14)
        ece = sum(float((bins == k).mean()) * abs(float(correct[bins == k].mean())
                   - float(confidence[bins == k].mean())) for k in range(15) if (bins == k).any())
        by_slot[slot] = dict(count=len(y), accuracy=float(correct.mean()),
                             nll=float(-logp[np.arange(len(y)), start + y[:, j]].mean()),
                             brier=float(np.square(brier).sum(1).mean()), ece=ece,
                             mean_confidence=float(confidence.mean()))
    macro = {key: float(np.mean([m[key] for m in by_slot.values()]))
             for key in ['accuracy', 'nll', 'brier', 'ece', 'mean_confidence']}
    macro['count'] = len(y)*6
    return dict(by_slot=by_slot, macro=macro,
                minimum_slot_accuracy=min(m['accuracy'] for m in by_slot.values()),
                whole_utterance_accuracy=float(np.stack(correct_slots, 1).all(1).mean())), p


def json_decision(logits, option_orders=None, temperatures=None):
    value = np.asarray(logits)
    if value.shape != (51,):
        raise ValueError('one native decision observation required')
    orders = OPTIONS if option_orders is None else option_orders
    if (set(orders) != set(OPTIONS) or any(len(orders[s]) != len(o)
            or set(orders[s]) != set(o) for s, o in OPTIONS.items())):
        raise ValueError('every native option exactly once is required')
    p = np.exp(_log_probabilities(value[None], temperatures))[0]
    result = {}
    for slot, start, options in zip(OPTIONS, BOUNDARIES[:-1], OPTIONS.values(), strict=True):
        # Reorder the already normalized probabilities, avoiding numerical order effects.
        result[slot] = {o: float(p[start + options.index(o)]) for o in orders[slot]}
    return json.dumps(result, allow_nan=False)


def selected_epoch(history):
    if not history or any(not np.isfinite(h['validation']['macro']['nll']) for h in history):
        raise ValueError('finite nonempty validation history required')
    return min(history, key=lambda h: (h['validation']['macro']['nll'], h['epoch']))['epoch']


def select_temperatures(cal_logits, cal_targets):
    y = _targets(cal_targets, len(cal_logits))
    losses = []
    grid = np.geomspace(.25, 4, 17)
    for t in grid:
        logp = _log_probabilities(cal_logits, np.full(6, t))
        losses.append([-logp[np.arange(len(y)), start+y[:, j]].mean()
                       for j, start in enumerate(BOUNDARIES[:-1])])
    return grid[np.asarray(losses).argmin(0)]
