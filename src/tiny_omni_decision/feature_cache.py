"""Small bounded LRU for immutable observation features."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class BoundedFeatureCache:
    """Cache content-addressed frozen features without retaining source media."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("feature cache capacity must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[Any, str]] = OrderedDict()

    def get(self, key: str) -> tuple[Any, str] | None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._entries[key] = entry
        return entry

    def put(self, key: str, feature: Any, feature_sha256: str) -> None:
        if not key or not feature_sha256:
            raise ValueError("feature cache key and hash must be nonempty")
        self._entries.pop(key, None)
        self._entries[key] = (feature, feature_sha256)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
