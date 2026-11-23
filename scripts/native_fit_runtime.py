"""Read-only feature transport for the common native model; no labels in inputs."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from scripts.native_candidate_inputs import positions
from tiny_omni_decision.native_candidate import NativeTokenBlock


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_text_feature_overlay(observations, banks, query_map, cache_root, expected_receipt_sha256):
    """Return copied Path A metadata pointing at a verified 768-D BERT cache."""
    root = Path(cache_root)
    result_path = root / "result.json"
    if _sha256(result_path) != expected_receipt_sha256:
        raise ValueError("BERT cache receipt SHA-256 differs")
    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_NATIVE_BERT_FEATURES_CACHE":
        raise ValueError("complete BERT feature cache required")
    for name, digest in receipt.get("artifact_hashes", {}).items():
        if _sha256(root / name) != digest:
            raise ValueError("BERT cache artifact changed: " + name)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    query_order = json.loads((root / "query-order.json").read_text(encoding="utf-8"))
    states = index.get("states", {})
    queries = index.get("queries", {})
    if (
        receipt.get("unique_text_states") != len(states)
        or receipt.get("unique_query_pairs") != len(queries)
        or len(query_order) != len(queries)
        or len(set(query_order)) != len(query_order)
        or set(query_order) != set(queries)
        or set(queries) != set(query_map)
    ):
        raise ValueError("BERT feature order or source identity coverage differs")

    replacement_banks = dict(banks)
    for name, shape in index["shapes"].items():
        overlay_name = "bert/" + name
        replacement_banks[overlay_name] = {
            "path": str(root / name),
            "sha256": receipt["artifact_hashes"][name],
            "shape": shape,
        }
    replacement_observations = {}
    text_ids = set()
    for ident, entry in observations.items():
        if entry.get("source") != "text":
            replacement_observations[ident] = entry
            continue
        text_ids.add(ident)
        key = ident.removeprefix("text:")
        if key not in states:
            raise ValueError("BERT cache omitted text observation: " + ident)
        meta = states[key]
        replacement_observations[ident] = {
            **entry,
            "features": [
                {
                    "kind": "text",
                    "bank": "bert/" + meta["file"],
                    "offset": meta["offset"],
                    "tokens": meta["tokens"],
                }
            ],
        }
    if text_ids != {"text:" + key for key in states}:
        raise ValueError("BERT state cache and text observations do not match exactly")

    replacement_queries = {
        key: {"bank": "bert/queries.npy", "offset": offset}
        for offset, key in enumerate(query_order)
    }
    return replacement_observations, replacement_banks, replacement_queries, 768


class FeatureStore:
    def __init__(self, observations, banks, query_map, device="cuda", *, text_width=384):
        if type(text_width) is not int or text_width < 1:
            raise ValueError("text_width must be a positive integer")
        self.text_width = text_width
        self.observations, self.banks, self.device = observations, banks, device
        self.arrays, self.cache = OrderedDict(), OrderedDict()
        self.cache_bytes = 0
        self.query_indices = {}
        vectors = []
        for key, entry in query_map.items():
            value = self.array(entry["bank"])[entry["offset"]]
            if (
                value.shape != (text_width,)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
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
                width = (
                    768 if feature["kind"] in ("image", "video")
                    else self.text_width if feature["kind"] == "text"
                    else 384
                )
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
