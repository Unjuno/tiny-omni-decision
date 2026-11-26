"""Acquire all409 fixed development AV clips, leaving official TEST and quarantine closed."""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from scripts.fetch_greatest_hits_train_inventory import write
from scripts.greatest_hits_media import planned_media
from scripts.greatest_hits_source import CurlRangeSource, member_bytes
from scripts.probe_greatest_hits_train import DIRECTORY_SHA, ETAG, SIZE, URL, sha

PARTITION_SHA = "6c9ac8eb8e12bd5dc0af77c8d162a40c753d5a215d07eb85dc3a69c3973d9f98"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    if (sha(args.directory) != DIRECTORY_SHA
            or sha(args.inventory / "partition.json") != PARTITION_SHA):
        raise ValueError("pinned source/grouping changed")
    inventory = json.loads((args.inventory / "result.json").read_text())
    if inventory["status"] != "PASS_FULL_OFFICIAL_TRAIN_INVENTORY":
        raise ValueError("complete annotation inventory required")
    plan = json.loads((args.inventory / "partition.json").read_text())
    entries = {e["name"]: e for e in json.loads(args.directory.read_text())}
    selected = planned_media(plan, entries)
    if len(selected) != 409 or sum(r["entry"]["bytes"] for r in selected) != 643_096_543:
        raise ValueError("registered media scope differs")
    args.output.mkdir(exist_ok=False)
    (args.output / "members").mkdir()
    write(args.output / "plan.json", selected)
    started, stop = time.monotonic(), threading.Event()
    records = {}
    result = dict(status="RUNNING", command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  started_utc=datetime.now(UTC).isoformat(), workers=2,
                  dispatch_deadline_seconds=1800, request_seconds=90,
                  partition_sha256=PARTITION_SHA, directory_sha256=DIRECTORY_SHA,
                  media_plan_sha256=sha(args.output / "plan.json"),
                  test_media_or_labels_opened=False, quarantine_media_opened=False,
                  model_loaded=False, optimizer_updates=0, sealed_audit_opened=False)

    def fetch(row):
        ident = row["parent"]
        if stop.is_set() or time.monotonic() - started > 1800:
            stop.set()
            return ident, dict(status="NOT_DISPATCHED", split=row["split"])
        source = None
        try:
            source = CurlRangeSource(URL, SIZE, ETAG, args.output / (ident + "-ranges"),
                                     request_seconds=90)
            data = member_bytes(row["entry"], source.read)
            name = ident + "_denoised.mp4"
            (args.output / "members" / name).write_bytes(data)
            return ident, dict(status="PASS", split=row["split"], file=name, bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest(),
                               requested_bytes=source.requested)
        except Exception as exc:
            stop.set()
            return ident, dict(status="FAILED", split=row["split"], error=repr(exc),
                               requested_bytes=source.requested if source else 0)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(fetch, row) for row in selected]
            for future in as_completed(pending):
                ident, record = future.result()
                records[ident] = record
                write(args.output / "progress.json", dict(completed=len(records),
                      passed=sum(r["status"] == "PASS" for r in records.values()),
                      last_id=ident, last_status=record["status"],
                      seconds=time.monotonic() - started))
        if len(records) != 409 or any(r["status"] != "PASS" for r in records.values()):
            raise RuntimeError("incomplete media acquisition; preserve partial receipts")
        if sha(args.directory) != DIRECTORY_SHA or sha(
                args.inventory / "partition.json") != PARTITION_SHA:
            raise ValueError("frozen inputs changed")
        result["status"] = "PASS_COMPLETE_DEVELOPMENT_MEDIA"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        write(args.output / "members.json", dict(sorted(records.items())))
        result.update(seconds=time.monotonic() - started,
                      finished_utc=datetime.now(UTC).isoformat(),
                      completed=len(records), passed=sum(r["status"] == "PASS"
                                                         for r in records.values()),
                      requested_bytes=sum(r.get("requested_bytes", 0) for r in records.values()),
                      source_hashes={p: sha(repo / p) for p in [
                          "scripts/greatest_hits_media.py", "scripts/greatest_hits_source.py",
                          "scripts/fetch_greatest_hits_development_media.py"]})
        write(args.output / "result.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
