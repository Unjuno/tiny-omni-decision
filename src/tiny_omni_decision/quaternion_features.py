"""Sign-invariant orientation representation; no frame calibration inferred."""

import numpy as np


def quaternion_rotation_matrices(quaternions):
    """Convert unit wxyz quaternions to active, column-vector rotation matrices.

    Normalizes only rounding error within 1e-3 norm tolerance. This conversion
    does not establish a dataset's world/body convention or remove mounting
    differences; callers must preserve sensor identities and metadata.
    """
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or not q.size or not np.isfinite(q).all():
        raise ValueError("finite nonempty wxyz quaternions required")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.allclose(norms, 1, atol=1e-3, rtol=0):
        raise ValueError("unit quaternions required")
    w, x, y, z = np.moveaxis(q / norms, -1, 0)
    entries = [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
               2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
               2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    return np.stack(entries, axis=-1).reshape(q.shape[:-1] + (3, 3))


def temporal_orientation_features(timestamps_ms, quaternions):
    """Eight equal-duration bins; retain every sample, including initial pose.

    Outputs arithmetic averages of matrix entries, not new rotation matrices.
    Empty bins fail instead of interpolating. No cross-modal clock inference.
    """
    times = np.asarray(timestamps_ms, dtype=np.float64)
    rotations = quaternion_rotation_matrices(quaternions)
    if (times.ndim != 1 or len(times) < 2 or rotations.ndim != 4
            or len(times) != len(rotations) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("aligned increasing times and sensor quaternion sequences required")
    bins = np.minimum(((times - times[0]) / (times[-1] - times[0]) * 8).astype(int), 7)
    if len(np.unique(bins)) != 8:
        raise ValueError("empty temporal bin")
    return np.stack([rotations[bins == i].mean(axis=0) for i in range(8)])
