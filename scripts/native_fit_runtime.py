"""Read-only feature transport for the common native model; no labels in inputs."""

from collections import OrderedDict

import numpy as np
import torch

from scripts.native_candidate_inputs import positions
from tiny_omni_decision.native_candidate import NativeTokenBlock


class FeatureStore:
    def __init__(self, observations, banks, query_map, device="cuda"):
        self.observations, self.banks, self.device = observations, banks, device
        self.arrays, self.cache = OrderedDict(), OrderedDict()
        self.cache_bytes = 0
        self.query_indices = {}
        vectors = []
        for key, entry in query_map.items():
            value = self.array(entry["bank"])[entry["offset"]]
            if value.shape != (384,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid saved query feature")
            self.query_indices[key] = len(vectors)
            vectors.append(value.copy())
        self.queries = torch.from_numpy(np.stack(vectors)).to(device)

    def array(self, name):
        if name not in self.arrays:
            self.arrays[name] = np.load(self.banks[name]["path"], mmap_mode="r", allow_pickle=False)
            while len(self.arrays) > 32:
                self.arrays.popitem(last=False)
        self.arrays.move_to_end(name)
        return self.arrays[name]

    def get(self, row):
        ident = row["observation"]
        if ident not in self.cache:
            blocks, size = [], 0
            for feature in self.observations[ident]["features"]:
                bank = self.array(feature["bank"])
                if "index" in feature:
                    bank = bank[feature["index"]]
                start, count = feature["offset"], feature["tokens"]
                value = np.array(bank[start : start + count], copy=True)
                width = 768 if feature["kind"] in ("image", "video") else 384
                if (
                    value.shape != (count, width)
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("invalid saved native feature slice")
                x = torch.from_numpy(value).to(self.device)
                p = positions(feature["kind"], count).to(self.device)
                mask = torch.ones(count, dtype=torch.bool, device=self.device)
                size += sum(v.numel() * v.element_size() for v in (x, p, mask))
                blocks.append(NativeTokenBlock(feature["kind"], x, p, mask))
            self.cache[ident] = (blocks, size)
            self.cache_bytes += size
            while self.cache_bytes > 64 * 2**20 and len(self.cache) > 1:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.cache_bytes -= old_size
        self.cache.move_to_end(ident)
        indices = torch.tensor(
            [self.query_indices[k] for k in row["queries"]], device=self.device, dtype=torch.long
        )
        return self.cache[ident][0], self.queries[indices]
