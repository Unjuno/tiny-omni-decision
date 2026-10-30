"""Strict scene grouping and source-free original QA input routing."""

import hashlib

import numpy as np


def scene_split(identities, media_hashes, *, held_out=20, seed=17):
    if (
        not isinstance(identities, list)
        or not isinstance(media_hashes, list)
        or len(identities) != len(media_hashes)
        or type(held_out) is not int
        or not 0 < held_out < len(identities)
        or type(seed) is not int
        or any(not isinstance(i, str) or not i for i in identities + media_hashes)
        or len(set(identities)) != len(identities)
        or len(set(media_hashes)) != len(media_hashes)
    ):
        raise ValueError("unique complete media identities/content and independent split required")
    ranking = sorted(
        range(len(identities)),
        key=lambda i: (
            hashlib.sha256(
                f"fine-qa:{seed}:{identities[i]}:{media_hashes[i]}".encode()
            ).hexdigest(),
            identities[i],
        ),
    )
    selected = set(ranking[:held_out])
    return (
        np.array([i for i in range(len(identities)) if i not in selected], np.int64),
        np.array([i for i in range(len(identities)) if i in selected], np.int64),
    )


def paired_subset(pairs, text, indices):
    if (
        not isinstance(pairs, list)
        or not pairs
        or not isinstance(text, np.ndarray)
        or text.shape != (2 * len(pairs), 3, 1536)
        or text.dtype != np.float32
        or not np.isfinite(text).all()
        or not isinstance(indices, np.ndarray)
        or indices.ndim != 1
        or indices.dtype != np.int64
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(pairs))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("complete paired native text and unique scene indices required")
    if len({p["video_id"] for p in pairs}) != len(pairs):
        raise ValueError("duplicate video identity")
    for p in pairs:
        if (
            p["split"] != "train"
            or len(p["options"]) != 3
            or len(set(p["options"])) != 3
            or [q["condition"] for q in p["questions"]] != ["initial", "final"]
            or any(
                type(q["target"]) is not int or q["target"] not in range(3) for q in p["questions"]
            )
        ):
            raise ValueError("original train source conditions and three options required")
    selected = [pairs[int(i)] for i in indices]
    rows = np.array([q for i in indices for q in (2 * int(i), 2 * int(i) + 1)], np.int64)
    return dict(
        pairs=selected,
        text=text[rows].copy(),
        targets=np.array([q["target"] for p in selected for q in p["questions"]], np.int64),
    )


def context_view(appearance, positions, control="normal"):
    if (
        not isinstance(appearance, np.ndarray)
        or not isinstance(positions, np.ndarray)
        or appearance.ndim != 3
        or min(appearance.shape[:2]) < 1
        or appearance.shape[-1] != 384
        or positions.shape != appearance.shape[:2] + (3,)
        or appearance.dtype != np.float32
        or positions.dtype != np.float32
        or not np.isfinite(appearance).all()
        or not np.isfinite(positions).all()
        or control not in {"normal", "zero", "initial", "reverse"}
    ):
        raise ValueError("aligned native spatial inputs and explicit control required")
    if control == "initial":
        appearance = np.broadcast_to(appearance[:1], appearance.shape)
    elif control == "reverse":
        appearance = appearance[::-1]
    result = np.concatenate((appearance, positions), axis=-1).reshape(-1, 387)
    if control == "zero":
        result.fill(0)
    return result
