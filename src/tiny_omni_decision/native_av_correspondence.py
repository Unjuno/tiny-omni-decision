"""Task-scoped frozen native AV matching with reusable content-bound states.

Caller-owned verified backends decode raw bytes and return native51 logits.
No targets, transcripts, sample identities or text encoder enter this readout.
"""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

BOUNDS = np.cumsum((0, 4, 4, 4, 25, 10, 4))


@dataclass(frozen=True)
class NativeAVState:
    owner: object
    modality: str
    profile_sha256: str
    media_sha256: str
    time_range: tuple[float, float]
    logits: tuple[float, ...]


def _logps(logits, temperatures):
    value = np.asarray(logits, np.float64)
    blocks = []
    for t, start, end in zip(temperatures, BOUNDS[:-1], BOUNDS[1:], strict=True):
        z = value[:, start:end] / t
        z -= z.max(1, keepdims=True)
        blocks.append(z - np.log(np.exp(z).sum(1, keepdims=True)))
    return np.concatenate(blocks, 1)


class NativeAVCorrespondence:
    """Fixed four-video choice and explicit uniform missing-evidence policy.

    This caches task-specific posteriors, not generic world-state semantics.
    Recreate the runtime after any backend/checkpoint/calibration/profile change.
    """

    def __init__(
        self,
        audio_backend,
        video_backend,
        audio_temperatures,
        video_temperatures,
        query_temperature,
        profile_sha256,
        *,
        max_entries=16,
    ):
        temps = [np.asarray(t, np.float64) for t in [audio_temperatures, video_temperatures]]
        if (
            any(t.shape != (6,) or not np.isfinite(t).all() or (t <= 0).any() for t in temps)
            or not np.isfinite(query_temperature)
            or query_temperature <= 0
            or not isinstance(profile_sha256, str)
            or len(profile_sha256) != 64
            or any(c not in "0123456789abcdef" for c in profile_sha256)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= 64
            or not callable(audio_backend)
            or not callable(video_backend)
        ):
            raise ValueError(
                "verified profile, positive fixed calibration and bounded backends required"
            )
        self._backends = {"audio": audio_backend, "video": video_backend}
        self._temps = [tuple(t.tolist()) for t in temps]
        self._temperature = float(query_temperature)
        self._profile = profile_sha256
        self._owner = object()
        self._cache = OrderedDict()
        self._max_entries = max_entries
        self.cache_hits = 0
        self.backend_calls = {"audio": 0, "video": 0}

    @property
    def cache_entries(self):
        return len(self._cache)

    def encode(self, modality, content, *, time_range, expected_sha256=None):
        times = np.asarray(time_range, np.float64)
        if (
            modality not in self._backends
            or not isinstance(content, bytes)
            or not 0 < len(content) <= 64 * 2**20
            or times.shape != (2,)
            or not np.isfinite(times).all()
            or times[0] < 0
            or times[1] <= times[0]
        ):
            raise ValueError("bounded raw media and finite ordered time range required")
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("media SHA256 differs")
        key = (self._profile, modality, digest, tuple(times.tolist()))
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        logits = np.asarray(self._backends[modality](content))
        self.backend_calls[modality] += 1
        if logits.shape != (51,) or not np.isfinite(logits).all():
            raise ValueError("complete finite native51 logits required; no missing fallback")
        state = NativeAVState(
            self._owner, modality, self._profile, digest, key[-1], tuple(logits.tolist())
        )
        self._cache[key] = state
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return state

    def _check(self, state, kind):
        if (
            not isinstance(state, NativeAVState)
            or state.owner is not self._owner
            or state.modality != kind
            or state.profile_sha256 != self._profile
        ):
            raise ValueError("state owner/modality/profile differs")

    def decide(self, audio, videos, option_ids):
        if (
            len(option_ids) != 4
            or any(not isinstance(i, str) or not i for i in option_ids)
            or len(set(option_ids)) != 4
        ):
            raise ValueError("four distinct opaque custom IDs required")
        if audio is not None:
            self._check(audio, "audio")
        if videos is not None:
            if len(videos) != 4:
                raise ValueError("four candidate states required")
            for v in videos:
                self._check(v, "video")
            if len({v.media_sha256 for v in videos}) != 4:
                raise ValueError("distinct video media required")
        ids = sorted(option_ids)
        scores = np.zeros(4, np.float64)
        if audio is not None and videos is not None:
            a = _logps([audio.logits], self._temps[0])
            ordered = [videos[option_ids.index(i)] for i in ids]
            v = _logps([s.logits for s in ordered], self._temps[1])
            for start, end in zip(BOUNDS[:-1], BOUNDS[1:], strict=True):
                joint = a[:, start:end][:, None, :] + v[None, :, start:end]
                maximum = joint.max(-1)
                scores += (maximum + np.log(np.exp(joint - maximum[:, :, None]).sum(-1)))[0]
        z = (scores / self._temperature)[None]
        z -= z.max(1, keepdims=True)
        p = np.exp(z - np.log(np.exp(z).sum(1, keepdims=True)))[0]
        return json.dumps(
            dict(
                selected_option_id=ids[int(p.argmax())],
                probabilities=dict(zip(ids, p.tolist(), strict=True)),
            ),
            sort_keys=True,
        )
