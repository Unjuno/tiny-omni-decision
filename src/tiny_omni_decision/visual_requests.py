"""Dependency-free bounded local raw-media requests for both worker environments."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualAsset:
    path: str
    sha256: str

    def __post_init__(self):
        if (not isinstance(self.path, str) or not self.path
                or not isinstance(self.sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)):
            raise ValueError("local path and lowercase SHA-256 required")


@dataclass(frozen=True)
class VisualRequest:
    key: str
    modality: str
    assets: tuple[VisualAsset, ...]

    def __post_init__(self):
        if (not isinstance(self.key, str) or not 1 <= len(self.key) <= 128
                or self.modality not in {"image", "video"}
                or not isinstance(self.assets, (list, tuple)) or not 1 <= len(self.assets) <= 8):
            raise ValueError("invalid visual request")
        assets = tuple(a if isinstance(a, VisualAsset) else VisualAsset(**a) for a in self.assets)
        object.__setattr__(self, "assets", assets)
        if self.modality == "video" and len(assets) != 1:
            raise ValueError("video requests require exactly one clip")

    def output_shape(self):
        return (len(self.assets), 17, 768) if self.modality == "image" else (1, 2048, 768)
