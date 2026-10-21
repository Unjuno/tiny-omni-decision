"""Decode every fixed development AV clip on CPU and account for every annotation window."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from scripts.fetch_greatest_hits_development_media import PARTITION_SHA
from scripts.fetch_greatest_hits_train_inventory import write
from scripts.greatest_hits_media import reject_cross_split_content, window_integrity
from scripts.probe_greatest_hits_train import sha

ANNOTATIONS_SHA = "24e8df777bee34102591e7b15901b931034ee2df3b549d632fb5b7094322e3dd"


def decode_one(path, parent, split, expected_sha, rows):
    import av
    import numpy as np

    started = time.monotonic()
    if sha(path) != expected_sha:
        raise ValueError("acquired media hash differs")
    with av.open(str(path)) as container:
        streams = [(s.index, s.type) for s in container.streams]
    if Counter(t for _, t in streams) != Counter({"audio": 1, "video": 1}):
        raise ValueError("one audio and one video stream required")
    result = dict(parent=parent, split=split, streams={})
    for index, kind in streams:
        digest = hashlib.sha256()
        count, first, previous, end = 0, None, None, None
        samples, energy, peak, max_gap = 0, 0.0, 0.0, 0.0
        with av.open(str(path)) as container:
            stream = container.streams[index]
            stream.thread_type = "NONE"
            info = dict(codec=stream.codec_context.name, time_base=str(stream.time_base))
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("missing stream PTS")
                stamp = float(frame.pts * frame.time_base)
                if previous is not None and stamp <= previous:
                    raise ValueError("nonmonotonic stream PTS")
                first = stamp if first is None else first
                if end is not None:
                    max_gap = max(max_gap, abs(stamp - end))
                previous, count = stamp, count + 1
                if kind == "video":
                    array = frame.to_ndarray(format="rgb24")
                    signature = [frame.width, frame.height, "rgb24"]
                    end = stamp + float(1 / stream.average_rate)
                    info.update(width=frame.width, height=frame.height,
                                average_rate=str(stream.average_rate),
                                end_estimator="last PTS + inverse average frame rate")
                else:
                    array = frame.to_ndarray()
                    signature = [frame.sample_rate, len(frame.layout.channels), str(array.dtype)]
                    if (not np.issubdtype(array.dtype, np.floating)
                            or not np.isfinite(array).all()):
                        raise ValueError("finite float PCM required")
                    samples += frame.samples
                    energy += float(np.sum(array.astype(np.float64) ** 2))
                    peak = max(peak, float(np.max(np.abs(array))))
                    end = stamp + frame.samples / frame.sample_rate
                    info.update(sample_rate=frame.sample_rate, channels=len(frame.layout.channels))
                if count == 1:
                    digest.update(json.dumps(signature).encode())
                    info["signature"] = signature
                elif signature != info["signature"]:
                    raise ValueError("stream format changes within parent")
                digest.update(array.tobytes(order="C"))
        if not count or (kind == "audio" and energy <= 0):
            raise ValueError("empty or silent stream")
        info.update(decoded_frames=count, first_pts=first, decoded_end=end,
                    max_abs_frame_gap=max_gap, strictly_increasing_pts=True)
        if kind == "audio":
            info.update(samples_per_channel=samples, peak=peak, all_finite=True,
                        rms=(energy / (samples * info["channels"])) ** 0.5)
        result["streams"][kind] = info
        result[kind + "_sha256"] = digest.hexdigest()
    result["windows"] = window_integrity(rows,
        (result["streams"]["audio"]["first_pts"], result["streams"]["audio"]["decoded_end"]),
        (result["streams"]["video"]["first_pts"], result["streams"]["video"]["decoded_end"]))
    if sha(path) != expected_sha:
        raise ValueError("source changed during decode")
    result["seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only process required")
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    import av
    import numpy as np

    if (sha(args.inventory / "annotations.json") != ANNOTATIONS_SHA
            or sha(args.inventory / "partition.json") != PARTITION_SHA):
        raise ValueError("frozen annotation/partition mismatch")
    annotations = json.loads((args.inventory / "annotations.json").read_text())
    plan = json.loads((args.inventory / "partition.json").read_text())
    media_receipt = json.loads((args.media / "result.json").read_text())
    if media_receipt["status"] != "PASS_COMPLETE_DEVELOPMENT_MEDIA":
        raise ValueError("complete acquisition required")
    members = json.loads((args.media / "members.json").read_text())
    owners = {ident: split for split in ("train", "validation") for ident in plan[split]}
    if set(members) != set(owners) or len(owners) != 409:
        raise ValueError("exact409 media scope required")
    args.output.mkdir(exist_ok=False)
    started, results, errors = time.monotonic(), {}, {}
    result = dict(status="RUNNING", command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  started_utc=datetime.now(UTC).isoformat(), python=platform.python_version(),
                  pyav=av.__version__, numpy=np.__version__, workers=4, model_loaded=False,
                  optimizer_updates=0, test_media_or_labels_opened=False, sealed_audit_opened=False,
                  annotations_sha256=ANNOTATIONS_SHA, partition_sha256=PARTITION_SHA,
                  media_receipt_sha256=sha(args.media / "result.json"))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(decode_one, args.media / "members" / (ident + "_denoised.mp4"),
                       ident, owners[ident], item["sha256"], annotations[ident]): ident
                       for ident, item in members.items()}
            for future in as_completed(pending):
                ident = pending[future]
                try:
                    results[ident] = future.result()
                    write(args.output / (ident + ".json"), results[ident])
                except Exception as exc:
                    errors[ident] = repr(exc)
                write(args.output / "progress.json", dict(passed=len(results), errors=errors,
                      seconds=time.monotonic() - started, last_parent=ident))
        if errors or len(results) != 409:
            raise ValueError("incomplete decode; retain every failing parent")
        reject_cross_split_content(list(results.values()))
        result["status"] = "PASS_COMPLETE_DECODE"
        if any(r["windows"]["out_of_bounds_windows"] for r in results.values()):
            result["status"] = "PASS_DECODE_WITH_WINDOW_ISSUES"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        result.update(seconds=time.monotonic() - started, completed=len(results), errors=errors,
                      finished_utc=datetime.now(UTC).isoformat(),
                      source_sha256=sha(Path(__file__)),
                      helper_sha256=sha(repo / "scripts/greatest_hits_media.py"),
                      parents_with_window_issues=sum(bool(r["windows"]["out_of_bounds_windows"])
                                                     for r in results.values()),
                      window_rows=sum(r["windows"]["rows"] for r in results.values()),
                      invalid_windows=sum(len(r["windows"]["out_of_bounds_windows"])
                                          for r in results.values()))
        write(args.output / "result.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
