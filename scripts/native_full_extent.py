"""A single full-extent bilinear/black-padding policy; historical preprocessing is unchanged."""

import importlib.metadata

import numpy as np

POLICY = "full-extent-bilinear-centered-black-pad-288-v1"


def parent_probe(rows, occurrences):
    """Return the fixed original-pool TRAIN rows in corpus order for both arms."""
    if (
        not rows
        or not occurrences
        or any(type(i) is not int or i < 0 or i >= len(rows) for i in occurrences)
        or any(row.get("split") != "train" or not row.get("id") for row in rows)
        or len({row["id"] for row in rows}) != len(rows)
    ):
        raise ValueError("unique TRAIN rows and valid parent occurrences required")
    return [rows[i] for i in sorted(set(occurrences))]


def encoder_distribution_versions(version=importlib.metadata.version):
    result = {name: version(name) for name in ("jax", "jaxlib", "numpy", "av", "psutil")}
    try:
        result["tensorflow"] = version("tensorflow")
    except importlib.metadata.PackageNotFoundError:
        result["tensorflow_cpu"] = version("tensorflow_cpu")
    return result


def full_extent_geometry(height, width, size=288):
    if any(type(v) is not int or v < 1 for v in (height, width, size)):
        raise ValueError("positive integer image/canvas dimensions required")
    scale = min(size / height, size / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    top, left = (size - resized_height) // 2, (size - resized_width) // 2
    return dict(
        source_height=height,
        source_width=width,
        canvas=size,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=top,
        pad_bottom=size - resized_height - top,
        pad_left=left,
        pad_right=size - resized_width - left,
        source_box=[0, 0, width, height],
        fill_rgb=[0, 0, 0],
        interpolation="bilinear",
        normalization="float32/255",
        caveat="Full spatial extent; reduced object scale and padding are part of policy.",
    )


def preprocess_full_extent(pixels, tf, size=288):
    rgb = np.asarray(pixels)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim not in (3, 4)
        or rgb.shape[-1] != 3
        or any(d < 1 for d in rgb.shape)
    ):
        raise ValueError("nonempty HWC or THWC RGB uint8 pixels required")
    geometry = full_extent_geometry(int(rgb.shape[-3]), int(rgb.shape[-2]), size)
    height, width = geometry["resized_height"], geometry["resized_width"]
    resized = (
        np.asarray(tf.image.resize(rgb, (height, width), method="bilinear"), dtype=np.float32)
        / 255.0
    )
    if (
        resized.shape != (*rgb.shape[:-3], height, width, 3)
        or not np.isfinite(resized).all()
        or resized.min() < 0
        or resized.max() > 1
    ):
        raise ValueError("invalid resized RGB tensor")
    result = np.zeros((*rgb.shape[:-3], size, size, 3), dtype=np.float32)
    top, left = geometry["pad_top"], geometry["pad_left"]
    result[..., top : top + height, left : left + width, :] = resized
    return result, geometry
