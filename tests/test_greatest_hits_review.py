import hashlib

import numpy as np
import pytest


def test_parent_per_date_is_fixed_independent_of_order():
    from scripts.greatest_hits_review import review_parents

    train = ['2015-01-01-00-00-01', '2015-01-01-00-00-02', '2015-01-02-00-00-01']
    expected = [min(train[:2], key=lambda p: hashlib.sha256(('17:' + p).encode()).hexdigest()),
                train[-1]]
    assert review_parents(train) == expected
    assert review_parents(train[::-1]) == expected
    with pytest.raises(ValueError):
        review_parents(train + train[:1])


def test_review_row_uses_first_complete_and_retains_all_missing_parent():
    from scripts.greatest_hits_review import review_event

    rows = [dict(seconds=1., fields=[None] * 3),
            dict(seconds=2., fields=['wood', 'hit', 'static']),
            dict(seconds=3., fields=['metal', 'hit', 'static'])]
    assert review_event(rows) == 1
    assert review_event(rows[:1]) == 0


def test_rms_offset_is_acoustic_not_video_alignment():
    from scripts.greatest_hits_review import acoustic_evidence

    pcm = np.zeros(48000, dtype=np.float32)
    pcm[24000:24960] = .25
    r = acoustic_evidence(pcm, 1.0, 1.25)
    assert len(r['rms_10ms']) == 50
    assert r['rms_10ms'][25] == .25
    assert r['max_rms_bin_offset_seconds'] == pytest.approx(.005)
    assert r['physical_av_offset'] == 'UNKNOWN'


@pytest.mark.parametrize('pcm', [np.zeros(1, np.float32), np.zeros(48000, np.float64),
                                np.full(48000, np.nan, np.float32)])
def test_invalid_audio_rejected(pcm):
    from scripts.greatest_hits_review import acoustic_evidence

    with pytest.raises(ValueError):
        acoustic_evidence(pcm, 0, .25)
