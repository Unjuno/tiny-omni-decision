"""Prepare frozen training/validation HARTH windows without accessing audit subjects."""

import csv
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.prepare_harth_native_smoke import ARCHIVE_SHA, CHANNELS, CODES, SOURCE, SPLIT_SHA
from scripts.range_generalization import sha, write
from tiny_omni_decision.sensor_windows import contiguous_windows


def prepare(output):
    archive = SOURCE / "harth.zip"
    splits_path = SOURCE / "frozen-subject-splits.json"
    if sha(archive) != ARCHIVE_SHA or sha(splits_path) != SPLIT_SHA:
        raise ValueError("source or frozen split hash mismatch")
    splits = json.loads(splits_path.read_text())["splits"]
    all_members = sum(splits.values(), [])
    if len(set(all_members)) != len(all_members):
        raise ValueError("subject overlap")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    records = []
    with zipfile.ZipFile(archive) as z:
        for split in ("train", "validation"):
            for member in splits[split]:
                raw = z.read(member)
                rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
                anchor = datetime.fromisoformat(rows[0]["timestamp"])
                ts = np.array([
                    (datetime.fromisoformat(r["timestamp"]) - anchor).total_seconds() * 1000
                    for r in rows
                ], dtype=np.float64)
                xs = np.array([[float(r[c]) for c in CHANNELS] for r in rows], dtype=np.float32)
                ys = np.array([int(r["label"]) for r in rows], dtype=np.int16)
                if not np.isfinite(xs).all() or not set(ys.tolist()).issubset(CODES):
                    raise ValueError("invalid source")
                windows = contiguous_windows(
                    ts.tolist(), [True] * len(ts), length=128, stride=64, max_gap_ms=30
                )
                if not windows:
                    raise ValueError("empty subject")
                starts = np.array([a for a, _ in windows])
                indices = starts[:, None] + np.arange(128)[None, :]
                subject = Path(member).stem
                target = ys[starts + 64]
                mixed = (ys[indices] != target[:, None]).any(1)
                path = output / f"{split}-{subject}.npz"
                np.savez_compressed(
                    path, values=xs[indices], timestamps_ms=ts[indices],
                    label_sequence=ys[indices], target=target, mixed=mixed,
                    source_intervals=np.array(windows),
                )
                row = {
                    "split": split, "subject": subject, "member": member,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_samples": len(rows), "windows": len(windows),
                    "mixed_windows": int(mixed.sum()), "path": str(path),
                    "sha256": sha(path),
                    "class_counts": {str(c): int((target == c).sum()) for c in CODES},
                }
                records.append(row)
                write(output / "progress.json", records)
                print(json.dumps({k: row[k] for k in ("split", "subject", "windows")}), flush=True)
    write(output / "result.json", {
        "archive_sha256": ARCHIVE_SHA, "split_sha256": SPLIT_SHA,
        "subjects": records, "seconds": time.perf_counter() - started,
        "held_out_contents_opened": ["validation"],
        "calibration_or_audit_opened": False,
    })


if __name__ == "__main__":
    prepare(Path("C:/CodexArtifacts/pretrained-reuse-r2/harth-subject-windows-01"))
