"""Deterministic recording correspondence manifests, not model feature inputs."""

import hashlib


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build_quartets(video_ids, *, split, limit, seed=17):
    if split not in ("train", "valid"):
        raise ValueError("development splits only")
    ids = list(video_ids)
    if len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("unique nonempty video IDs required")
    if type(limit) is not int or limit <= 0 or limit % 4 or limit > len(ids):
        raise ValueError("available whole quartets required")
    ordered = sorted(ids, key=lambda v: key(f"{seed}:{v}"))[:limit]
    result = []
    for offset in range(0, limit, 4):
        videos = ordered[offset : offset + 4]
        quartet = f"{split}:q{offset // 4:04}"
        positions = sorted(range(4), key=lambda p: key(f"{seed}:{quartet}:position:{p}"))
        for query, correct_position in zip(videos, positions, strict=True):
            candidates = sorted(
                (v for v in videos if v != query), key=lambda v: key(f"{seed}:{query}:{v}")
            )
            candidates.insert(correct_position, query)
            result.append(
                dict(
                    record_id=f"{quartet}:{query}",
                    quartet_id=quartet,
                    split=split,
                    audio_video_id=query,
                    candidate_video_ids=candidates,
                    target_index=correct_position,
                    warning="IDs are provenance/routing only, never scorer inputs",
                )
            )
    return result
