"""Explicit full-field image resize policy for the isolated crop comparison."""

import numpy as np


def whole_image_rgb(image, resize):
    source = np.asarray(image)
    if (source.ndim != 3 or source.shape[-1] != 3 or min(source.shape[:2]) < 1
            or source.dtype != np.uint8):
        raise ValueError("nonempty uint8 RGB image required")
    result = np.asarray(resize(source, (288, 288), method="bilinear"), dtype=np.float32) / 255
    if (result.shape != (288, 288, 3) or not np.isfinite(result).all()
            or result.min() < 0 or result.max() > 1):
        raise ValueError("invalid resized RGB values")
    return result
