"""Label-blind whole-stream coverage and validation-only AV retention selection."""

import hashlib
import json
import math
from collections import defaultdict, deque

MODALITIES = ('text', 'image', 'audio', 'video')
MAX_STEPS = 32768


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def _row(row):
    target = row.get('target', row.get('target_index'))
    options = row.get('options', [])
    if (not isinstance(row.get('id'), str) or not row['id'] or row.get('split') != 'train'
            or not isinstance(options, list) or not 2 <= len(options) <= 40
            or any(not isinstance(o, str) or not o.strip() for o in options)
            or len(set(options)) != len(options) or type(target) is not int
            or not 0 <= target < len(options)
            or ('target' in row and 'target_index' in row and row['target_index'] != target)):
        raise ValueError('unique TRAIN identity, complete options and valid target required')
    return target


class _Stream:
    def __init__(self, name, rows, keys):
        self.name, self.cycle, self.draws = name, -1, 0
        self.buckets = defaultdict(list)
        for row in rows:
            key = tuple(row.get(k) for k in keys)
            if any(not isinstance(k, str) or not k for k in key):
                raise ValueError('explicit source/cohort/task bucket keys required')
            self.buckets[key].append(row['id'])
        self.pending = deque()
        self.seen, self.exhausted = set(), []

    def take(self):
        if not self.pending:
            self.cycle += 1
            self.queues = {k: deque(sorted(ids, key=lambda ident: (
                digest([17, self.name, self.cycle, k, ident]), ident)))
                for k, ids in sorted(self.buckets.items())}
            self.pending = deque(self.queues)
        key = self.pending.popleft()
        ident = self.queues[key].popleft()
        self.draws += 1
        self.seen.add(ident)
        if self.queues[key]:
            self.pending.append(key)
        else:
            self.exhausted.append(dict(cycle=self.cycle, after_draw=self.draws, bucket=list(key)))
        return ident

    def summary(self):
        return dict(consumed=self.draws, unique_questions=len(self.seen),
                    repeated_questions=self.draws-len(self.seen), cycles_started=self.cycle+1,
                    pool_questions=sum(map(len, self.buckets.values())),
                    exhausted_buckets=self.exhausted)


def build_plan(joint, common, steps=MAX_STEPS):
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS or not joint or not common:
        raise ValueError('positive at-most32768 budget and nonempty TRAIN streams required')
    rows = [*joint, *common]
    for row in rows:
        _row(row)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('unique identities across both TRAIN corpora required')
    if {r.get('modality') for r in common} != set(MODALITIES):
        raise ValueError('all and only four existing modalities required')
    streams = {'joint': _Stream('joint', joint, ('task',))}
    streams.update({m: _Stream(m, [r for r in common if r['modality'] == m],
                              ('source', 'cohort', 'task')) for m in MODALITIES})
    order = [[*[streams['joint'].take() for _ in range(4)],
              *[streams[m].take() for m in MODALITIES]] for _ in range(steps)]
    return dict(seed=17, steps=steps, order=order, order_sha256=digest(order),
                streams={k: v.summary() for k, v in streams.items()})


def presentation(row, occurrence):
    target = _row(row)
    if type(occurrence) is not int or occurrence < 0:
        raise ValueError('nonnegative integer presentation index required')
    order = sorted(range(len(row['options'])),
                   key=lambda i: digest([17, occurrence, row['id'], i]))
    return dict(order=order, target=order.index(target))


def _number(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def select_checkpoint(best, joint_nll, retention, initial):
    if not _number(joint_nll) or (best is not None and not _number(best)):
        raise ValueError('finite nonnegative joint NLL required')
    for metrics in (retention, initial):
        if (set(metrics.get('modality_macro', {})) != set(MODALITIES)
                or not _number(metrics.get('macro', {}).get('nll'))
                or any(not _number(v.get('accuracy')) or v['accuracy'] > 1
                       for v in metrics['modality_macro'].values())):
            raise ValueError('complete finite four-modality retention metrics required')
    reasons = [m+'_accuracy_drop_over_2pp' for m in MODALITIES
               if retention['modality_macro'][m]['accuracy']
               < initial['modality_macro'][m]['accuracy']-.02-1e-12]
    if retention['macro']['nll'] > initial['macro']['nll']+1e-5:
        reasons.append('existing_macro_nll_worse_than_initial')
    eligible = not reasons
    return dict(eligible=eligible, selected=eligible and (best is None or joint_nll < best-1e-5),
                score=joint_nll, reasons=reasons)
