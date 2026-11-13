"""Explicit0.5s event inputs; old mono16/24kHz frontend remains unchanged."""

import math
from fractions import Fraction

import numpy as np


def crop_audio(pcm, sample_rate, first_pts, onset):
    if (not isinstance(pcm, np.ndarray) or pcm.ndim != 1 or pcm.dtype != np.float32
            or not len(pcm) or not np.isfinite(pcm).all() or type(sample_rate) is not int
            or sample_rate != 96000 or not all(math.isfinite(t) for t in (first_pts, onset))):
        raise ValueError("finite native mono96kHz float32 PCM and timestamps required")
    center = round((onset - first_pts) * sample_rate)
    left, right = center - 24000, center + 24000
    if left < 0 or right > len(pcm):
        raise ValueError("complete unpadded0.5s event window required")
    info = dict(first_sample=left, end_sample_exclusive=right,
                start_seconds=first_pts + left / sample_rate,
                end_seconds=first_pts + right / sample_rate,
                center_error_seconds=first_pts + center / sample_rate - onset,
                rounding="nearest sample; ties-to-even")
    if abs(info["center_error_seconds"]) > 0.5 / sample_rate + 1e-12:
        raise ValueError("event timing error exceeds half sample")
    return pcm[left:right].copy(), info


def select_video(pts, start, end):
    if (not isinstance(pts, np.ndarray) or pts.ndim != 1 or len(pts) < 8
            or not np.isfinite(pts).all() or not (np.diff(pts) > 0).all()
            or not all(math.isfinite(t) for t in (start, end)) or start >= end):
        raise ValueError("finite increasing video PTS and valid interval required")
    inside = np.flatnonzero((pts >= start) & (pts < end))
    if len(inside) < 8:
        raise ValueError("eight distinct real frames required; no padding/repetition")
    return [int(inside[round(i * (len(inside) - 1) / 7)]) for i in range(8)]


def resample_direct(pcm, target_rate):
    if (not isinstance(pcm, np.ndarray) or pcm.shape != (48000,) or pcm.dtype != np.float32
            or not np.isfinite(pcm).all() or type(target_rate) is not int
            or target_rate not in (16000, 32000)):
        raise ValueError("native0.5s96kHz float32 and explicit16/32kHz target required")
    import av

    frame = av.AudioFrame.from_ndarray(pcm[None].copy(), format="fltp", layout="mono")
    frame.sample_rate, frame.pts, frame.time_base = 96000, 0, Fraction(1, 96000)
    converter = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
    frames = converter.resample(frame) + converter.resample(None)
    result = np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
    if (result.shape != (target_rate // 2,) or result.dtype != np.float32
            or not np.isfinite(result).all()):
        raise ValueError("finite, exact half-second direct resampling required")
    return result
