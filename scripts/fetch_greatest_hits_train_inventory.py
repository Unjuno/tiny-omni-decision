"""Fetch only official TRAIN annotations, after freezing date-grouped development IDs."""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from scripts.greatest_hits_inventory import account_annotations, partition_train
from scripts.greatest_hits_source import CurlRangeSource, member_bytes, parse_ids, parse_times
from scripts.probe_greatest_hits_train import DIRECTORY_SHA, ETAG, SIZE, URL, sha

TRAIN_SHA = "b01681a47c23f1e3ede19517e566241302ca0bba0df54fd29b16050d781d8088"
TEST_SHA = "162cce800a7cc2d8efe247f2ab1877918772d98542a451c2eac342c795c043eb"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    inputs = {args.directory: DIRECTORY_SHA, args.probe / "train.txt": TRAIN_SHA,
              args.probe / "test.txt": TEST_SHA}
    if any(sha(p) != expected for p, expected in inputs.items()):
        raise ValueError("frozen directory/split metadata changed")
    train, test = [parse_ids((args.probe / (s + ".txt")).read_bytes()) for s in ("train", "test")]
    lookup = {e["name"]: e for e in json.loads(args.directory.read_text())}
    entries = {i: lookup["vis-data-256/" + i + "_times.txt"] for i in train}
    worst_bytes = sum(30 + 512 + 4096 + e["compressed"] for e in entries.values())
    if (worst_bytes > 64 * 2**20 or any(not 0 < e["bytes"] <= 64 * 1024
                                      or not 0 < e["compressed"] <= 64 * 1024
                                      for e in entries.values())):
        raise ValueError("annotation/aggregate resource cap exceeded")
    plan = partition_train(train, test)
    if {s: len(ids) for s, ids in plan.items()} != dict(train=308, validation=101, quarantine=324):
        raise ValueError("prospective partition differs")
    args.output.mkdir(exist_ok=False)
    (args.output / "members").mkdir()
    # This file is written before any new annotation payload is read or acquired.
    write(args.output / "partition.json", plan)
    write(args.output / "partition-hashes.json", {
        s: hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest() for s, ids in plan.items()})
    old = json.loads((args.probe / "result.json").read_text())
    started, stop = time.monotonic(), threading.Event()
    records, annotations = {}, {}
    result = dict(status="RUNNING", command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  started_utc=datetime.now(UTC).isoformat(), workers=8,
                  dispatch_deadline_seconds=900, worst_case_transfer_bytes=worst_bytes,
                  test_media_or_labels_opened=False, model_loaded=False, optimizer_updates=0,
                  sealed_audit_opened=False, partition_sha256=sha(args.output / "partition.json"))

    def fetch(ident):
        if stop.is_set() or time.monotonic() - started > 900:
            stop.set()
            return ident, dict(status="NOT_DISPATCHED"), None
        name = ident + "_times.txt"
        source = None
        try:
            if name in old["fetched"]:
                data = (args.probe / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != old["fetched"][name]["sha256"]:
                    raise ValueError("reused source differs")
            else:
                source = CurlRangeSource(URL, SIZE, ETAG, args.output / (ident + "-ranges"))
                data = member_bytes(entries[ident], source.read)
            if (len(data) != entries[ident]["bytes"]
                    or f"{zlib.crc32(data):08x}" != entries[ident]["crc32"]):
                raise ValueError("source label length/CRC mismatch")
            parsed = parse_times(data)
            (args.output / "members" / name).write_bytes(data)
            return ident, dict(status="PASS", reused=source is None, bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest(),
                               requested_bytes=source.requested if source else 0), parsed
        except Exception as exc:
            stop.set()
            return ident, dict(status="FAILED", error=repr(exc),
                               requested_bytes=source.requested if source else 0), None

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = [pool.submit(fetch, ident) for ident in sorted(train)]
            for future in as_completed(pending):
                ident, record, parsed = future.result()
                records[ident] = record
                if parsed is not None:
                    annotations[ident] = parsed
                write(args.output / "progress.json", dict(completed=len(records),
                      parsed=len(annotations), last_id=ident, last_status=record["status"],
                      seconds=time.monotonic() - started))
        if len(annotations) != len(train):
            raise RuntimeError(f"incomplete annotation inventory: {len(annotations)}/{len(train)}")
        if any(sha(p) != expected for p, expected in inputs.items()):
            raise ValueError("frozen input changed during acquisition")
        write(args.output / "annotations.json", dict(sorted(annotations.items())))
        write(args.output / "inventory.json", account_annotations(plan, annotations))
        result["status"] = "PASS_FULL_OFFICIAL_TRAIN_INVENTORY"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        write(args.output / "members.json", dict(sorted(records.items())))
        result.update(finished_utc=datetime.now(UTC).isoformat(),
                      seconds=time.monotonic() - started,
                      completed=len(records), parsed=len(annotations),
                      requested_bytes=sum(r.get("requested_bytes", 0) for r in records.values()),
                      source_hashes={p: sha(repo / p) for p in [
                          "scripts/greatest_hits_inventory.py", "scripts/greatest_hits_source.py",
                          "scripts/fetch_greatest_hits_train_inventory.py"]})
        write(args.output / "result.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
