"""Fixed date-coverage TRAIN inspection; acoustic offsets do not certify AV alignment."""

import hashlib
import math

import numpy as np

from scripts.greatest_hits_source import ID


def review_parents(train):
    if not train or len(train) != len(set(train)) or any(not ID.fullmatch(p) for p in train):
        raise ValueError('unique valid TRAIN parent IDs required')
    return [min((p for p in train if p[:10] == date),
                key=lambda p: hashlib.sha256(('17:' + p).encode()).hexdigest())
            for date in sorted({p[:10] for p in train})]


def review_event(rows):
    if not rows or any(len(r['fields']) != 3 for r in rows):
        raise ValueError('released three-field event rows required')
    return next((i for i, row in enumerate(rows) if None not in row['fields']), 0)


def acoustic_evidence(pcm, start, onset):
    if (not isinstance(pcm, np.ndarray) or pcm.shape != (48000,) or pcm.dtype != np.float32
            or not np.isfinite(pcm).all() or not all(math.isfinite(x) for x in (start, onset))):
        raise ValueError('finite native0.5s mono96kHz window and times required')
    rms = np.sqrt(np.square(pcm.astype(np.float64).reshape(50, 960)).mean(1))
    strongest = int(rms.argmax())
    return dict(rms_10ms=rms.tolist(), max_rms_bin=strongest,
                max_rms_bin_offset_seconds=start + (strongest + .5) * .01 - onset,
                physical_av_offset='UNKNOWN')
