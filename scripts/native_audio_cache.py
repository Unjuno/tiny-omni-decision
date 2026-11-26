"""Pure storage checks for the fixed complete native audio feature profile."""

import numpy as np


def validate_profile(config):
    required = dict(
        seed=17,
        batch_size=1,
        dtype="float32",
        attention="eager",
        max_examples=7410,
        expected_tensor_bytes=1908900864,
        max_wall_seconds=3600,
        max_vram_bytes=3 * 2**30,
        max_rss_bytes=6 * 2**30,
        disk_reserve_bytes=2 * 2**30,
    )
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("registered complete audio cache profile differs")


def store_features(banks, row, values, phase):
    if (
        set(values) != {"speech", "event"}
        or row["split"] not in {"train", "validation"}
        or phase not in {"cache", "replay"}
    ):
        raise ValueError("both audio paths and explicit development storage phase required")
    maximum = 0.0
    for kind, value in values.items():
        expected_shape = (row["speech_tokens"], 384) if kind == "speech" else (1, 384)
        offset = row[kind + "_offset"]
        bank = banks[row["split"] + "-" + kind + ".npy"]
        if (
            value.dtype != np.float32
            or value.shape != expected_shape
            or not np.isfinite(value).all()
            or bank.dtype != np.float32
            or bank.ndim != 2
            or bank.shape[1] != 384
            or offset < 0
            or offset + len(value) > len(bank)
        ):
            raise ValueError("audio feature/offset/bank format differs")
        target = bank[offset : offset + len(value)]
        if phase == "cache":
            target[:] = value
        elif phase == "replay":
            delta = float(np.max(np.abs(target - value)))
            maximum = max(maximum, delta)
            if not np.array_equal(target, value):
                raise ValueError("full fresh audio feature replay differs")
        else:
            raise ValueError("explicit cache or replay required")
    return maximum
