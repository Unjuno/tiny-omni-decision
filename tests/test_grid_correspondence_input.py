import json
from dataclasses import asdict

import numpy as np
import pytest

from tiny_omni_decision.grid_correspondence_input import build_input
from tiny_omni_decision.native_av_correspondence import NativeAVCorrespondence


def fixture():
    codes = ['a', 'b', 'c', 'd']
    ids = ['opaque-1', 'opaque-2', 'opaque-3', 'opaque-4']
    row = {'codes': codes, 'option_ids': ids, 'speaker': 'private',
           'queries': [{'audio_code': c, 'target_id': i}
                       for c, i in zip(codes, ids, strict=True)]}
    audio = {c: f'audio-{c}'.encode() for c in codes}
    video = {c: f'video-{c}'.encode() for c in codes}
    return row, audio, video


def test_public_input_excludes_private_fields_and_balances_roles():
    row, audio, video = fixture()
    requests_and_targets = [build_input(row, audio, video, i) for i in range(4)]
    assert {target for _, target in requests_and_targets} == set(row['option_ids'])
    for public, _ in requests_and_targets:
        assert set(asdict(public)) == {'audio', 'videos', 'option_ids'}
        assert public.videos == tuple(video[c] for c in row['codes'])


def test_reject_wrong_label_and_duplicate_video():
    row, audio, video = fixture()
    row['queries'][0]['target_id'] = 'opaque-2'
    with pytest.raises(ValueError, match='mapping'):
        build_input(row, audio, video, 0)
    row, audio, video = fixture()
    video['b'] = video['a']
    with pytest.raises(ValueError, match='duplicate video'):
        build_input(row, audio, video, 0)


def test_public_input_connects_to_existing_json_runtime():
    row, audio, video = fixture()
    public, _ = build_input(row, audio, video, 0)
    runtime = NativeAVCorrespondence(lambda _: np.zeros(51), lambda _: np.zeros(51),
                                     [1] * 6, [1] * 6, 1.0, '0' * 64)
    audio_state = runtime.encode('audio', public.audio, time_range=(0, 1))
    videos = [runtime.encode('video', v, time_range=(0, 1)) for v in public.videos]
    decision = json.loads(runtime.decide(audio_state, videos, public.option_ids))
    assert set(decision['probabilities']) == set(public.option_ids)
    assert sum(decision['probabilities'].values()) == pytest.approx(1.0)
