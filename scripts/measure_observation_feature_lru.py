"""Measure a bounded in-memory feature LRU on a fixed validation query order."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tiny_omni_decision.cache import (
    InMemoryObservationFeatureLRU,
    ObservationFeatureCache,
    ObservationFeatureKey,
)
from tiny_omni_decision.dataset import sha256_file


def _key_from_dict(value: dict[str, Any]) -> ObservationFeatureKey:
    fields = dict(value)
    fields.pop("schema_version", None)
    fields["feature_shape"] = tuple(fields["feature_shape"])
    return ObservationFeatureKey(**fields)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--capacity-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing measurement: {args.output}")
    cache_manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.validation_jsonl.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if cache_manifest.get("schema_version") != 1 or not rows:
        raise ValueError("unsupported cache manifest or empty validation query sequence")
    by_observation: dict[str, tuple[ObservationFeatureKey, str]] = {}
    for entry in cache_manifest.get("entries", []):
        key = _key_from_dict(entry["key"])
        if key.observation_sha256 != entry.get("observation_sha256"):
            raise ValueError("cache manifest observation identity mismatch")
        by_observation[key.observation_sha256] = (key, entry["feature_payload_sha256"])

    memory = InMemoryObservationFeatureLRU(args.capacity_bytes)
    persistent = ObservationFeatureCache(args.cache_dir)
    persistent_reads = 0
    cache_misses = 0
    evictions_after_query: list[int] = []
    peak_payload_bytes = 0
    started = time.perf_counter()
    query_order = []
    for row in rows:
        observation_sha = str(row["media_sha256"])
        item = by_observation.get(observation_sha)
        if item is None:
            raise ValueError(f"no persisted observation feature for {observation_sha}")
        key, expected_payload_sha = item
        query_order.append(str(row["id"]))
        if memory.get(key) is None:
            cache_misses += 1
            cached = persistent.get(key)
            if cached.payload_sha256 != expected_payload_sha:
                raise ValueError(f"persistent feature hash mismatch for {observation_sha}")
            persistent_reads += 1
            if not memory.put(key, cached.payload):
                raise ValueError("feature payload exceeds configured in-memory budget")
        peak_payload_bytes = max(peak_payload_bytes, memory.bytes_used)
        evictions_after_query.append(memory.eviction_count)
    elapsed = time.perf_counter() - started

    report = {
        "schema_version": 1,
        "status": "complete_in_memory_feature_lru_measurement",
        "scope": "fixed CLEVRER validation query order; payload-byte accounting only",
        "capacity_bytes": memory.max_bytes,
        "query_count": len(rows),
        "unique_observations": len({str(row["media_sha256"]) for row in rows}),
        "query_order_sha256": hashlib.sha256(
            "\n".join(query_order).encode("utf-8")
        ).hexdigest(),
        "query_ids": query_order,
        "memory_hits": memory.hit_count,
        "memory_misses": cache_misses,
        "persistent_cache_reads": persistent_reads,
        "evictions": memory.eviction_count,
        "evictions_after_each_query": evictions_after_query,
        "resident_entries_at_end": memory.entry_count,
        "resident_payload_bytes_at_end": memory.bytes_used,
        "peak_resident_payload_bytes": peak_payload_bytes,
        "elapsed_seconds": elapsed,
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "validation_jsonl_sha256": sha256_file(args.validation_jsonl),
        "persistent_entry_count": len(cache_manifest["entries"]),
        "note": (
            "The configured limit excludes Python object/container overhead. This replays "
            "already-extracted feature payloads and is not an inference or "
            "encoder-latency benchmark."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
