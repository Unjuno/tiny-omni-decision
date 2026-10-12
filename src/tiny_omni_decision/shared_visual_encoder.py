"""Fixed image/video feature profiles sharing one explicit VideoPrism tree."""

import numpy as np

from tiny_omni_decision.videoprism_spatial_features import spatial_cells


class SharedVisualEncoder:
    def __init__(self, model, variables, *, compile_fn):
        self.variables = variables
        self._spatial = compile_fn(lambda params, x: model.apply(
            params, x, train=False,
            return_intermediate={"spatial_features"})[1]["spatial_features"])
        self._temporal = compile_fn(lambda params, x: model.apply(params, x, train=False)[0])

    def encode(self, frames, *, modality):
        if modality not in {"image", "video"}:
            raise ValueError("supported modalities: image, video")
        x = np.asarray(frames)
        expected_frames = 1 if modality == "image" else 8
        if (x.ndim != 5 or x.shape[0] < 1 or x.shape[1:] != (expected_frames, 288, 288, 3)
                or x.dtype != np.float32 or not np.isfinite(x).all()
                or x.min() < 0 or x.max() > 1):
            raise ValueError("requires normalized float32 [B,T,288,288,3] for the fixed profile")
        if modality == "image":
            return spatial_cells(np.asarray(self._spatial(self.variables, x)))
        features = np.asarray(self._temporal(self.variables, x))
        if features.shape != (len(x), 2048, 768) or not np.isfinite(features).all():
            raise ValueError("invalid temporal features")
        return features.reshape(len(x), 8, 256, 768).mean(2)
