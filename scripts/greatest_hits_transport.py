"""Pure fail-closed binding of complete event caches; no labels in model inputs."""

import math

import numpy as np

from scripts.greatest_hits_audio_query import audio_positions, query_inventory
from scripts.greatest_hits_cache import frozen_events
from scripts.greatest_hits_features import event_positions
from scripts.native_language_cache import pair_key


def _clock(clock, onset):
    keys = {'start_seconds', 'end_seconds', 'first_sample', 'end_sample_exclusive',
            'center_error_seconds', 'rounding'}
    if set(clock) - keys or not keys-{'rounding'} <= set(clock):
        raise ValueError('explicit native audio clock required')
    start, end = clock['start_seconds'], clock['end_seconds']
    left, right = clock['first_sample'], clock['end_sample_exclusive']
    error = clock['center_error_seconds']
    if (not all(math.isfinite(v) for v in (start, end, error))
            or type(left) is not int or type(right) is not int or left < 0
            or right-left != 48000 or not math.isclose(end-start, .5, abs_tol=1e-9, rel_tol=0)
            or abs(error) > .5/96000+1e-12
            or not math.isclose((start+end)/2-onset, error, abs_tol=1e-9, rel_tol=0)):
        raise ValueError('invalid half-second nearest-sample clock')


def bind_caches(splits, audio, video, audio_plan, video_plan, *, phase):
    if phase not in ('full', 'smoke'):
        raise ValueError('explicit full/smoke scope required')
    expected = frozen_events(splits)
    queries = query_inventory(splits)
    selected = audio_plan['events']
    if (not selected or selected != video_plan or audio_plan['queries'] != queries
            or phase == 'full' and selected != expected):
        raise ValueError('audio/video plans differ from frozen scope')
    ids = {e['event_id'] for e in selected}
    if selected != [e for e in expected if e['event_id'] in ids] or (
            phase == 'smoke' and any(e['split'] != 'train' for e in selected)):
        raise ValueError('unique ordered declared TRAIN smoke or complete full events required')
    expected_by_id = {e['event_id']: e for e in selected}
    counts = dict(events=len(selected), parents=len({e['parent'] for e in selected}),
                  questions=sum(e['questions'] for e in selected))
    bound = []
    for receipt, status in ((audio, 'PASS_COMPLETE_AUDIO_QUERY_CACHE'),
                            (video, 'PASS_COMPLETE_VIDEO_CACHE')):
        if (receipt['status'] != status or receipt['phase'] != phase
                or any(receipt[f'{prefix}_{key}'] != n for prefix in ('expected', 'completed')
                       for key, n in counts.items())):
            raise ValueError('complete passing same-scope receipts required')
        records, parents = {}, set()
        for parent in receipt['parents']:
            if parent['parent'] in parents or not parent['events']:
                raise ValueError('unique nonempty cache parent required')
            parents.add(parent['parent'])
            for index, record in enumerate(parent['events']):
                ident = record['event_id']
                original = expected_by_id.get(ident)
                if (ident in records or original is None
                        or any(record.get(k) != v for k, v in original.items())
                        or record['parent'] != parent['parent']
                        or record['split'] != parent['split']
                        or type(record['bank_index']) is not int or record['bank_index'] != index):
                    raise ValueError('cache event identity/order/ownership differs')
                _clock(record['timing'], record['event_seconds'])
                records[ident] = dict(**original, bank_index=index,
                    bank_events=len(parent['events']), timing=dict(record['timing']))
        if list(records) != [e['event_id'] for e in selected] or len(parents) != counts['parents']:
            raise ValueError('incomplete or reordered cache binding')
        bound.append(records)
    if bound[0] != bound[1]:
        raise ValueError('audio/video physical clocks or event maps differ')
    questions = {row['id']: dict(event_id=row['event_id'], options=list(row['options']),
                    query_keys=[pair_key(row['question'], option) for option in row['options']])
                 for rows in splits.values() for row in rows if row['event_id'] in ids}
    if len(questions) != counts['questions']:
        raise ValueError('incomplete question binding')
    result = dict(events=bound[0], questions=questions,
                  query_keys=[q['key'] for q in queries], phase=phase)
    if 'audio_admission' in audio:
        result['audio_admission'] = dict(audio['audio_admission'])
    return result


def validate_positions(positions, pts, clock):
    if set(positions) != {'speech', 'event_audio', 'video'}:
        raise ValueError('all explicit positions required')
    try:
        expected = dict(**audio_positions(), video=event_positions(
            pts, clock['start_seconds'], clock['end_seconds'])['video'])
        for kind, value in expected.items():
            actual = positions[kind]
            if not isinstance(actual, np.ndarray) or actual.dtype != np.float32:
                raise ValueError('saved FP32 positions required')
            np.testing.assert_array_equal(actual, value)
    except (AssertionError, TypeError, KeyError) as exc:
        raise ValueError('saved geometry differs from explicit event clock') from exc
