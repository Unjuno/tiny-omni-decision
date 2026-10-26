"""Bind native Decision inputs to trusted extraction receipts and immutable caches.

Bindings belong to the application/extractor configuration, not a user request.
They certify a handoff from that trusted producer; hashes do not prove that an
untrusted producer actually ran its claimed encoder.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import ObservationFeatureCache, ObservationFeatureKey
from .observations import Observation


@dataclass(frozen=True)
class NativeFeatureBinding:
    key: ObservationFeatureKey
    feature_payload_sha256: str
    source_id: str
    clock_id: str
    timestamps_ms: tuple[float, ...]

    def __post_init__(self):
        if self.key.modality not in {"audio", "image", "video"}:
            raise ValueError("native cached features support audio, image and video")
        if self.key.feature_dtype != "float32":
            raise ValueError("native cache payload must be little-endian float32")
        if len(self.feature_payload_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.feature_payload_sha256
        ):
            raise ValueError("feature payload requires a lowercase SHA-256")
        if not self.source_id.strip() or not self.clock_id.strip():
            raise ValueError("binding source and clock must be nonempty")
        if (
            not isinstance(self.timestamps_ms, tuple)
            or not self.timestamps_ms
            or any(not math.isfinite(t) for t in self.timestamps_ms)
            or any(b <= a for a, b in zip(self.timestamps_ms, self.timestamps_ms[1:], strict=False))
        ):
            raise ValueError("binding timestamps must be finite and strictly increasing")
        if self.key.modality == "video" and len(self.timestamps_ms) != self.key.feature_shape[0]:
            raise ValueError("video binding timestamp count must equal feature sequence length")


def _check_observation(observation, binding):
    observation = Observation.model_validate(observation.model_dump(mode="json"))
    if (
        observation.modality != binding.key.modality
        or observation.source_id != binding.source_id
        or observation.payload_sha256 != binding.key.observation_sha256
        or observation.clock_id != binding.clock_id
        or observation.timestamps_ms != binding.timestamps_ms
    ):
        raise ValueError("observation metadata differs from trusted feature binding")
    if observation.payload_index is not None:
        raise ValueError("indexed media containers require an explicit payload-index binding")
    return observation


def load_bound_feature(
    observation: Observation,
    binding: NativeFeatureBinding,
    cache: ObservationFeatureCache,
) -> np.ndarray:
    """Verify actual local media and a producer-pinned cache payload on every read.

    Stat timestamps are intentionally not used to skip hashing. Indexed source
    containers need an explicit extraction contract and are currently rejected.
    Return an independent writable array, never a mutable view of cached state.
    """
    observation = _check_observation(observation, binding)
    if not all(observation.valid_mask):
        raise ValueError("incomplete observation cannot load a complete cached feature")
    digest = hashlib.sha256()
    with Path(observation.payload_ref).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != binding.key.observation_sha256:
        raise ValueError("actual media SHA-256 differs from trusted feature binding")
    entry = cache.get(binding.key)
    if entry.payload_sha256 != binding.feature_payload_sha256:
        raise ValueError("cached feature SHA-256 differs from trusted extraction receipt")
    if len(entry.payload) != math.prod(binding.key.feature_shape) * 4:
        raise ValueError("feature payload byte count does not match declared shape")
    values = np.frombuffer(entry.payload, dtype="<f4").reshape(binding.key.feature_shape)
    if not np.isfinite(values).all():
        raise ValueError("bound feature payload must be finite")
    return values.copy()


def decide_bound_native(
    readout,
    batch,
    binding: NativeFeatureBinding,
    cache: ObservationFeatureCache,
    *,
    taxonomy=None,
    program_tokens=None,
    temperature=1.0,
) -> dict:
    """Score trusted cached features with an existing native readout.

    The application supplies the producer binding independently of the request.
    The original tensor-level adapters remain available as low-level operations;
    this entry point adds actual-media, full-key and feature-digest verification.
    """
    import torch

    from .native_decision_json import native_classification_request, native_decision_json
    from .textfree_audio import TextFreeAudioReadout
    from .textfree_image import TextFreeImageReadout
    from .textfree_video import TextFreeVideoReadout

    modality = binding.key.modality
    types = {
        "audio": TextFreeAudioReadout,
        "image": TextFreeImageReadout,
        "video": TextFreeVideoReadout,
    }
    if not isinstance(readout, types[modality]) or readout.training:
        raise ValueError("bound native runtime requires a matching readout in eval mode")
    batch, _, abstain_reason = native_classification_request(batch, modality)
    observation = _check_observation(batch.observations[0], binding)
    if (
        modality == "audio"
        and (taxonomy is not None or program_tokens is not None)
        or modality == "image"
        and (program_tokens is not None or temperature != 1.0)
        or modality == "video"
        and temperature != 1.0
    ):
        raise ValueError("unsupported native readout arguments")
    if abstain_reason is not None:
        return native_decision_json(batch, None, abstain_reason=abstain_reason)
    values = load_bound_feature(observation, binding, cache)
    parameter = next(readout.parameters())
    features = torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)
    if modality == "audio":
        result = readout.decide_native(features, batch, temperature=temperature)
    elif modality == "image":
        result = readout.decide_native(features, taxonomy, batch)
    else:
        result = readout.decide_native(
            features, batch, taxonomy=taxonomy, program_tokens=program_tokens
        )
    return {
        **result,
        "feature_provenance": {
            "cache_id": binding.key.cache_id,
            "media_sha256": binding.key.observation_sha256,
            "feature_payload_sha256": binding.feature_payload_sha256,
        },
    }
