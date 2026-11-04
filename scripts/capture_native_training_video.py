"""Persist all456 existing development video token arrays; no fitting/evaluation."""

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import psutil

from scripts.native_candidate_inputs import require_source_revision
from scripts.native_video_cache import records_for_cache, sha, store_tokens
from tiny_omni_decision.token_transport import read_tokens

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
CONFIG = REPO / "configs/pretrained_reuse/native_video_cache_v1.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO):
        raise ValueError("clean committed source required")
    config = read(CONFIG)
    for path, expected in config["pins"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned source changed: {path}")
    source = Path(config["visual_source"])
    require_source_revision(source, config["visual_revision"], "videoprism")
    sample = Path(config["sample_dir"])
    splits = {s: [json.loads(line) for line in (sample / (s + ".jsonl"))
                  .read_text(encoding="utf-8").splitlines()] for s in ("train", "validation")}
    order = read(Path(config["request"]))
    rows = records_for_cache(splits, order, read(Path(config["historical_tokens"])))
    counts = dict(Counter(r["split"] for r in rows))
    if counts != {"train": 256, "validation": 200}:
        raise ValueError("full unchanged256/200 source cohort required")
    for row in rows:
        path = sample / row["media_path"]
        if path.stat().st_size != row["media_bytes"] or sha(path) != row["media_sha256"]:
            raise ValueError("native media changed")
    needed = len(rows) * (2048 * 768 * 4 + 128)
    if shutil.disk_usage(args.output.parent).free < needed + config["minimum_disk_bytes"]:
        raise RuntimeError("insufficient space for complete cache plus reserve")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    receipt = dict(status="RUNNING", started_utc=datetime.now(UTC).isoformat(),
                   source_commit=subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   config_sha256=sha(CONFIG), config=config, counts=counts,
                   model_updates=0, quality_evaluation=False, sealed_audit_loaded=False,
                   calibration_loaded=False, pid=os.getpid(), completed=0,
                   expected_payload_bytes=needed, cloud_cost=0, rows=[])
    write(args.output / "effective-config.json", config)
    write(args.output / "result.json", receipt)
    command = [config["worker_python"], "-m", "scripts.audit_token_ram_transport",
               "--audit", config["audit"], "--request", config["request"], "--worker", "--live"]
    receipt["command"] = sys.argv
    receipt["worker_command"] = command
    messages = queue.Queue(maxsize=2)
    child = None
    peak_rss = 0
    try:
        with (args.output / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=log)
            receipt["worker_pid"] = child.pid

            def receive():
                try:
                    while True:
                        item = read_tokens(child.stdout, (8, 16, 16, 768))
                        messages.put(item)
                        if item is None:
                            return
                except BaseException as exc:
                    messages.put(exc)

            thread = threading.Thread(target=receive, daemon=True)
            thread.start()
            while True:
                if time.perf_counter() - started > config["max_seconds"]:
                    raise RuntimeError("declared wall budget exhausted; cache incomplete")
                if shutil.disk_usage(args.output).free < config["minimum_disk_bytes"]:
                    raise RuntimeError("disk reserve reached; cache incomplete")
                try:
                    processes = [psutil.Process(), *psutil.Process().children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    peak_rss = max(peak_rss, rss)
                    if rss > config["max_rss_bytes"]:
                        raise RuntimeError("declared process-tree memory budget exhausted")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                try:
                    item = messages.get(timeout=5)
                except queue.Empty:
                    continue  # an observation timeout never restarts an encoder
                if isinstance(item, BaseException):
                    raise item
                if item is None:
                    break
                index = len(receipt["rows"])
                if index >= len(rows) or item[0] != str(index):
                    raise ValueError("worker stream order/count differs")
                receipt["rows"].append(store_tokens(args.output, rows[index], item[1]))
                receipt.update(completed=index + 1, seconds=time.perf_counter() - started,
                               observed_peak_process_tree_rss_bytes=peak_rss)
                write(args.output / "result.json", receipt)
                if index % 32 == 0 or index + 1 == len(rows):
                    print(json.dumps({k: receipt[k] for k in ["completed", "seconds"]}), flush=True)
            thread.join(timeout=10)
            if (thread.is_alive() or child.wait(timeout=30) != 0
                    or len(receipt["rows"]) != len(rows)):
                raise RuntimeError("incomplete or failed extraction")
        for path, expected in config["pins"].items():
            if sha(Path(path)) != expected:
                raise ValueError("source changed during extraction")
        receipt.update(status="PASS_COMPLETE_NATIVE_VIDEO_TOKEN_CACHE", seconds=time.perf_counter()
                       - started, finished_utc=datetime.now(UTC).isoformat(),
                       total_token_file_bytes=sum(r["file_bytes"] for r in receipt["rows"]),
                       memory_measurement="process tree sampled on token receipt or5s timeout; "
                       "may miss short peaks", all_historical_token_hashes_exact=True)
    except BaseException:
        receipt.update(status="INCOMPLETE", error=traceback.format_exc(),
                       seconds=time.perf_counter() - started)
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                # Only the dedicated child launched by this invocation, never a legacy job.
                for descendant in psutil.Process(child.pid).children(recursive=True):
                    descendant.terminate()
                child.terminate()
                child.wait(timeout=30)
            child.stdout.close()
        write(args.output / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["status", "completed", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
