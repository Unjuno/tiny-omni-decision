"""Strict media-group binding and immutable FP32 token cache publication."""

import hashlib
import re
from pathlib import PurePosixPath

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def records_for_cache(splits, order, prior):
    if set(splits) != {"train", "validation"} or any(not rows for rows in splits.values()):
        raise ValueError("nonempty train/validation only; no audit or calibration")
    media, owners = {}, {}
    for split, rows in splits.items():
        for row in rows:
            fields = ("media_sha256", "video_filename", "media_path", "media_bytes")
            if (row.get("split") != split or any(k not in row for k in fields)
                    or not re.fullmatch("[a-f0-9]{64}", row["media_sha256"])
                    or type(row["media_bytes"]) is not int or row["media_bytes"] <= 0):
                raise ValueError("complete fixed source identity required")
            path = PurePosixPath(row["media_path"])
            if path.is_absolute() or ".." in path.parts or path.name != row["video_filename"]:
                raise ValueError("relative native path/filename required")
            key = row["media_sha256"]
            record = {k: row[k] for k in fields}
            record["split"] = split
            if key in media and media[key] != record:
                raise ValueError("same media has inconsistent identity/size/split")
            media[key] = record
            for kind in fields[:3]:
                identity = (kind, row[kind])
                if identity in owners and owners[identity] != (split, key):
                    raise ValueError("underlying media/scene/path overlap or conflict")
                owners[identity] = (split, key)
    if len(order) != len(set(order)) or set(order) != set(media):
        raise ValueError("complete original scene order required; no additions/removals")
    legacy = {r["media_sha256"]: r["token_sha256"] for r in prior}
    if len(legacy) != len(prior) or set(legacy) != set(media) or any(
        not re.fullmatch("[a-f0-9]{64}", h) for h in legacy.values()
    ):
        raise ValueError("one historical raw token digest for every scene required")
    return [{**media[key], "token_sha256": legacy[key]} for key in order]


def store_tokens(output, row, tokens):
    value = np.asarray(tokens)
    if (value.shape != (8, 16, 16, 768) or value.dtype != np.float32
            or not np.isfinite(value).all()
            or hashlib.sha256(value.tobytes()).hexdigest() != row["token_sha256"]
            or not re.fullmatch("[a-f0-9]{64}", row["media_sha256"])):
        raise ValueError("exact historical FP32 eight-frame tokens required")
    path = output / (row["media_sha256"] + ".npy")
    with path.open("xb") as stream:
        np.save(stream, value.reshape(2048, 768), allow_pickle=False)
    reloaded = np.load(path, allow_pickle=False)
    if not np.array_equal(reloaded, value.reshape(2048, 768)):
        raise ValueError("token save/reload mismatch")
    return {**row, "file": path.name, "file_sha256": sha(path),
            "file_bytes": path.stat().st_size, "shape": [2048, 768],
            "dtype": "float32", "reloaded_exact": True}
