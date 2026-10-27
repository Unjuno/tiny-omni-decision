"""Audit instruction availability at past-only DROID training windows."""

import argparse
import json
import runpy
from collections import Counter
from pathlib import Path

from scripts.audit_droid_train_all_transitions import DATA, PARSER, PARSER_SHA, TRAIN
from scripts.audit_droid_train_temporal import RECEIPT_SHA, TRAIN_SHA, canonical_id, digest

REPO = Path(__file__).resolve().parents[1]
FIELDS = tuple(f"steps/language_instruction{suffix}" for suffix in ("", "_2", "_3"))


def availability(sequences, steps):
    """Count past-index presence without using target or future text."""
    if steps < 8 or any(len(seq) != steps for seq in sequences):
        raise ValueError("instruction sequence length mismatch")
    result = Counter()
    for target in range(8, steps):
        now = any(seq[target - 1].strip() for seq in sequences)
        ever = any(value.strip() for seq in sequences for value in seq[:target])
        result["windows"] += 1
        result["at_previous_step"] += bool(now)
        result["at_any_past_step"] += bool(ever)
    result["episode_any"] = int(any(v.strip() for seq in sequences for v in seq))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output required")
    receipt = DATA / "download-receipt.json"
    pinned = {receipt: RECEIPT_SHA, TRAIN: TRAIN_SHA, PARSER: PARSER_SHA}
    if any(digest(p) != h for p, h in pinned.items()):
        raise ValueError("pinned input mismatch")
    rows = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != 275 or len(by_id) != 275 or any(r["split"] != "train" for r in rows):
        raise ValueError("unexpected train cohort")
    api = runpy.run_path(str(PARSER), run_name="pinned_droid_parser")
    counts, seen, shards = Counter(), set(), 0
    for item in json.loads(receipt.read_bytes())["objects"]:
        if ".tfrecord-" not in item["name"]:
            continue
        path = DATA / item["name"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("shard hash mismatch")
        for raw in api["iter_tfrecord"](path):
            ctx = api["episode_example_context"](raw)
            identifier = canonical_id(ctx["episode_metadata/file_path"])
            if identifier not in by_id:
                continue
            row = by_id[identifier]
            group = canonical_id(ctx["episode_metadata/recording_folderpath"])
            if identifier in seen or group != row["group_id"]:
                raise ValueError("duplicate or mismatched recording")
            seqs = [api["episode_example_bytes_values"](raw, field) for field in FIELDS]
            result = availability(seqs, int(row["source_step_count"]))
            counts.update(result)
            seen.add(identifier)
        shards += 1
        print(f"checked {shards} shards; {len(seen)} train recordings", flush=True)
    if len(seen) != 275 or counts["windows"] != 80998:
        raise ValueError("cohort or window count mismatch")
    if any(digest(p) != h for p, h in pinned.items()):
        raise ValueError("input changed during audit")
    result = {"counts": dict(counts), "recordings": len(seen), "shards": shards,
              "input_hashes": {str(p): h for p, h in pinned.items()},
              "script_sha256": digest(Path(__file__)), "sealed_audit_loaded": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
