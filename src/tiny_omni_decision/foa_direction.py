"""Fixed broadband FOA direction diagnostic using documented W,Y,Z,X order."""

import numpy as np


def foa_direction(audio):
    """Unit [X,Y,Z] from mean W*directional-channel products, or None at zero.

    No filtering, fitting, event classification or uncertainty calibration.
    Assumes the STARSS23 ideal channel response convention, not arbitrary audio
    formats. Reflections, simultaneous sources and interference can bias it.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty four-channel audio required")
    products = (x[:, :1] * x[:, [3, 1, 2]]).mean(0)
    norm = np.linalg.norm(products)
    if not np.isfinite(norm):
        raise ValueError("direction arithmetic overflow")
    return products / norm if norm > 0 else None
