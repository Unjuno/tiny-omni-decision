"""Native audio/provenance helpers; transcripts never establish clock or truth."""

import hashlib
import json
import re

import numpy as np


def read_bound_json(path, expected):
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("immutable reference hash required before parsing")
    return json.loads(body)


def recheck(pins):
    from scripts.diagnose_robomnist_lengths import sha

    if any(sha(p) != value for p, value in pins.items()):
        raise ValueError("complete source/product/report pin changed")


def resample_native(pcm, sample_rate):
    if (
        sample_rate != 44100
        or pcm.dtype != np.float32
        or pcm.ndim != 2
        or pcm.shape[1] != 1
        or not 441 <= len(pcm) <= 44100 * 30
        or not np.isfinite(pcm).all()
    ):
        raise ValueError("bounded finite native mono44.1kHz float32 PCM required")
    import av

    frame = av.AudioFrame.from_ndarray(pcm.T.copy(), format="flt", layout="mono")
    frame.sample_rate = 44100
    frame.pts = 0
    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    frames = resampler.resample(frame) + resampler.resample(None)
    result = np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
    if (
        result.dtype != np.float32
        or not np.isfinite(result).all()
        or abs(len(result) / 16000 - len(pcm) / 44100) > 1 / 16000
    ):
        raise ValueError("actual complete resampling duration/count/precision differs")
    return result


def require_sequence(ids):
    if (
        ids.ndim != 1
        or ids.dtype != np.int64
        or not 5 <= len(ids) <= 4 + 128
        or ids[:4].tolist() != [50258, 50259, 50359, 50363]
        or ids[-1] != 50257
    ):
        raise ValueError("exact English prefix/terminal EOS within fixed cap required")


def describe_numbers(text):
    words = (
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    ).split()
    mapping = {word: i for i, word in enumerate(words)}
    values = []
    for word in re.findall(r"\b\w+\b", text.lower()):
        number = mapping.get(word)
        if word.isascii() and word.isdecimal() and int(word) <= 20:
            number = int(word)
        if number is not None:
            values.append(number)
    return dict(
        observations_0_to_20=values,
        speech_ground_truth="UNVERIFIED",
        collection_counter_active="UNKNOWN",
        common_clock="UNVERIFIED",
    )
