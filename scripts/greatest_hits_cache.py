"""Bind every frozen native question to its asset event; stream overlapping frames."""

import math
import re
from collections import deque

import numpy as np

from scripts.greatest_hits_source import ID
from scripts.greatest_hits_tasks import TASKS


def frozen_events(splits):
    if set(splits) != {'train', 'validation'} or any(not x for x in splits.values()):
        raise ValueError('nonempty development train/validation only')
    events, seen, owners = {}, set(), {}
    for split in ('train', 'validation'):
        for row in splits[split]:
            parent, kind, t = row['parent'], row['task'], row['event_seconds']
            line = row['source_line']
            if (not ID.fullmatch(parent) or kind not in TASKS or row['split'] != split
                    or type(line) is not int or line < 1 or not math.isfinite(t) or t < .25
                    or row['date_group'] != parent[:10]
                    or not re.fullmatch('[a-f0-9]{64}', row['asset_sha256'])
                    or row['media_file'] != parent + '_denoised.mp4'
                    or row['event_id'] != f'greatest-hits/{parent}/{line-1:04d}'
                    or row['id'] != row['event_id'] + '/' + kind or row['id'] in seen
                    or row['question'] != TASKS[kind][0] or row['options'] != TASKS[kind][1]
                    or type(row['target']) is not int
                    or not 0 <= row['target'] < len(row['options'])
                    or row['event_window_seconds'] != [t-.25, t+.25]
                    or row['modalities'] != ['audio', 'video']):
                raise ValueError('invalid or changed frozen question')
            seen.add(row['id'])
            for key, owner in [(('date', parent[:10]), split),
                               (('parent', parent), (split, row['asset_sha256'])),
                               (('asset', row['asset_sha256']), (split, parent))]:
                if key in owners and owners[key] != owner:
                    raise ValueError('cross-split identity or inconsistent asset ownership')
                owners[key] = owner
            fields = ('event_id', 'parent', 'split', 'date_group', 'asset_sha256',
                      'media_file', 'event_seconds', 'source_line')
            item = {k: row[k] for k in fields}
            prior = events.setdefault(row['event_id'], dict(**item, questions=0))
            if any(prior[k] != v for k, v in item.items()):
                raise ValueError('same event has inconsistent physical identity')
            prior['questions'] += 1
    return sorted(events.values(), key=lambda x: (x['split'] != 'train',
                                                  x['parent'], x['source_line']))


def frame_windows(frames, windows):
    if (not windows or any(not all(math.isfinite(x) for x in pair)
            or not math.isclose(pair[1]-pair[0], .5, rel_tol=0, abs_tol=1e-9)
            for pair in windows)
            or any(a[0] >= b[0] for a, b in zip(windows, windows[1:], strict=False))):
        raise ValueError('strictly ordered half-second windows required')
    iterator, retained, previous = iter(frames), deque(), -math.inf

    def take():
        nonlocal previous
        value = next(iterator, None)
        if value is not None:
            if not math.isfinite(value[0]) or value[0] <= previous:
                raise ValueError('finite strictly increasing actual frame PTS required')
            previous = value[0]
        return value

    pending = take()
    for start, end in windows:
        while pending is not None and pending[0] < end:
            if pending[0] >= start:
                retained.append(pending)
            pending = take()
        while retained and retained[0][0] < start:
            retained.popleft()
        if not 8 <= len(retained) <= 16:
            raise ValueError('complete eight-to-sixteen real event frames required')
        yield np.stack([x[1] for x in retained]), np.asarray([x[0] for x in retained], np.float64)
