"""Bounded, checksummed FP32 tensor frames for a separate frozen encoder process."""

import hashlib
import json
import math

import numpy as np


def write_tokens(stream, key, values):
    values = np.asarray(values)
    if values.dtype != np.float32 or not np.isfinite(values).all() or not key:
        raise ValueError("expected finite FP32 and nonempty asset key")
    payload = values.astype("<f4", copy=False).tobytes(order="C")
    header = (
        json.dumps(
            dict(
                key=key,
                shape=list(values.shape),
                dtype="<f4",
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        ).encode()
        + b"\n"
    )
    if len(header) > 4096:
        raise ValueError("oversized header")
    stream.write(header)
    stream.write(payload)
    stream.flush()


def read_tokens(stream, expected_shape):
    header = stream.readline(4097)
    if not header:
        return None
    if len(header) > 4096 or not header.endswith(b"\n"):
        raise ValueError("invalid frame header")
    record = json.loads(header)
    if (
        record.get("shape") != list(expected_shape)
        or record.get("dtype") != "<f4"
        or not isinstance(record.get("key"), str)
        or not record["key"]
    ):
        raise ValueError("unexpected tensor metadata")
    remaining = math.prod(expected_shape) * 4
    payload = bytearray()
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError("truncated tensor")
        payload.extend(chunk)
        remaining -= len(chunk)
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ValueError("tensor checksum mismatch")
    values = np.frombuffer(payload, dtype="<f4").reshape(expected_shape)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite tensor")
    return record["key"], values
