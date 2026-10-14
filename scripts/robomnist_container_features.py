"""Pure container/whole-lens/canonical-array helpers; no model imports."""

import hashlib
import struct
from pathlib import Path

import numpy as np


def append_free_atom(original, derived, payload_bytes=2**20):
    original, derived = Path(original), Path(derived)
    if original.resolve() == derived.resolve():
        raise ValueError("original source must remain protected")
    if derived.exists():
        raise FileExistsError(derived)
    if type(payload_bytes) is not int or not 0 < payload_bytes <= 2**20:
        raise ValueError("bounded positive free-space payload required")
    header = struct.pack(">I4s", payload_bytes + 8, b"free")
    h, prefix, count = hashlib.sha256(), hashlib.sha256(), 0
    with original.open("rb") as src, derived.open("xb") as dst:
        for b in iter(lambda: src.read(2**20), b""):
            count += len(b)
            prefix.update(b)
            h.update(b)
            dst.write(b)
        h.update(header)
        dst.write(header)
        payload = bytes(payload_bytes)
        h.update(payload)
        dst.write(payload)
    return dict(
        source_bytes=count,
        source_sha256=prefix.hexdigest(),
        derived_bytes=count + payload_bytes + 8,
        derived_sha256=h.hexdigest(),
        free_atom_bytes=payload_bytes + 8,
        payload_bytes=payload_bytes,
    )


def stereo_tiles(rgb, image_mean):
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or not 0 < rgb.shape[0] <= 2160
        or not 2 <= rgb.shape[1] <= 4096
        or rgb.shape[1] % 2
    ):
        raise ValueError("native bounded uint8 RGB and two equal horizontal lenses required")
    mean = np.asarray(image_mean, dtype=np.float64)
    if mean.shape != (3,) or not np.isfinite(mean).all() or (mean < 0).any() or (mean > 1).any():
        raise ValueError("finite native three-channel image mean required")
    from PIL import Image

    fill = np.rint(mean * 255).astype(np.uint8)
    halves = np.split(rgb, 2, axis=1)
    tiles = []
    for lens in halves:
        height, width = lens.shape[:2]
        scale = 224 / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = np.asarray(Image.fromarray(lens).resize(size, Image.Resampling.BICUBIC))
        canvas = np.empty((224, 224, 3), np.uint8)
        canvas[:] = fill
        x, y = (224 - size[0]) // 2, (224 - size[1]) // 2
        canvas[y : y + size[1], x : x + size[0]] = resized
        tiles.append(canvas)
    return np.stack(tiles)


def digest_arrays(arrays):
    h = hashlib.sha256()
    for name, a in sorted(arrays.items()):
        if not isinstance(name, str) or not isinstance(a, np.ndarray) or a.dtype.hasobject:
            raise ValueError("named native non-object arrays required")
        h.update(repr((name, a.shape, str(a.dtype))).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def same_arrays(first, second):
    if not first or set(first) != set(second):
        raise ValueError("all corresponding array keys required")
    for key, a in first.items():
        b = second[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise ValueError("array identity requires exact shape/dtype/every byte")
