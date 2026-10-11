"""Acquire only frozen STARSS23 development split media, with bounded disk and HTTP use."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.fetch_starss23_smoke import ARCHIVES, BASE, HttpRangeReader, archive_index


def load_splits(directory, summary):
    rows = []
    for split in ("train", "calibration", "validation"):
        path = directory / f"{split}.jsonl"
        if sha256_file(path) != summary["splits"][split]["sha256"]:
            raise ValueError("frozen split hash mismatch")
        part = [json.loads(line) for line in path.read_text().splitlines()]
        if any(r["split"] != split or r["upstream_split"] != "train" for r in part):
            raise ValueError("incorrect split or upstream evaluation data")
        rows.extend(part)
    if len({r["clip"] for r in rows}) != len(rows):
        raise ValueError("recording overlap")
    for room in {r["room"] for r in rows}:
        if len({r["split"] for r in rows if r["room"] == room}) != 1:
            raise ValueError("room overlap")
    return rows


def check_capacity(required, free, reserve=2 * 1024**3):
    if required + reserve > free:
        raise ValueError(
            f"insufficient disk: need {required} bytes plus {reserve} reserve; free {free}"
        )


def write_json(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reuse-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = load_splits(args.splits, summary)
    reuse = json.loads(args.reuse_probe.read_text())
    if reuse["status"] != "PASS" or reuse["record"] != summary["source_revision"]:
        raise ValueError("reuse source mismatch")
    existing = {r["clip"]: r for r in reuse["pairs"]}
    for row in existing.values():
        for media in row["files"].values():
            if sha256_file(Path(media["path"])) != media["sha256"]:
                raise ValueError("reuse media hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = {
        "status": "INDEXING",
        "pairs": [],
        "whole_archive_md5_verified": False,
        "dev_test_accessed": False,
        "summary_sha256": sha256_file(args.summary),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    started = time.perf_counter()
    readers = {
        name: HttpRangeReader(f"{BASE}/{name}/content", size, budget=5 * 1024**3)
        for name, (size, _) in ARCHIVES.items()
    }
    try:
        with (
            zipfile.ZipFile(readers["foa_dev.zip"]) as audio,
            zipfile.ZipFile(readers["video_dev.zip"]) as video,
        ):
            indices = {"audio": archive_index(audio, ".wav"), "video": archive_index(video, ".mp4")}
            archives = {"audio": audio, "video": video}
            required = compressed = 0
            for row in rows:
                for kind in indices:
                    entry = indices[kind][row["clip"]]
                    if "/dev-train-" not in entry.filename:
                        raise ValueError("media member outside official dev-train")
                    if entry.file_size > 512 * 1024**2:
                        raise ValueError("member exceeds 512 MiB safety bound")
                    if row["clip"] not in existing:
                        required += entry.file_size
                        compressed += entry.compress_size
            plan = {
                "recordings": len(rows),
                "reuse_recordings": len(existing),
                "new_uncompressed_bytes": required,
                "new_compressed_bytes": compressed,
                "free_disk_bytes": shutil.disk_usage(args.output_dir).free,
                "reserve_bytes": 2 * 1024**3,
                "splits": summary["splits"],
            }
            write_json(args.output_dir / "plan.json", plan)
            check_capacity(required, plan["free_disk_bytes"])
            progress["status"] = "PREFLIGHT_PASS"
            write_json(args.output_dir / "progress.json", progress)
            print(json.dumps(plan), flush=True)
            if not args.execute:
                return
            progress["status"] = "FETCHING"
            for row in rows:
                pair = {**row, "files": {}}
                for kind in indices:
                    entry = indices[kind][row["clip"]]
                    if row["clip"] in existing:
                        pair["files"][kind] = {
                            **existing[row["clip"]]["files"][kind],
                            "reused": True,
                        }
                    else:
                        check_capacity(entry.file_size, shutil.disk_usage(args.output_dir).free)
                        data = archives[kind].read(entry)  # ZIP CRC verified by zipfile.
                        path = args.output_dir / Path(entry.filename).name
                        with path.open("xb") as stream:
                            stream.write(data)
                        del data
                        pair["files"][kind] = {
                            "path": str(path.resolve()),
                            "sha256": sha256_file(path),
                            "bytes": entry.file_size,
                            "crc32": entry.CRC,
                            "member": entry.filename,
                            "reused": False,
                        }
                    progress["current"] = {"clip": row["clip"], "completed_media": kind}
                    write_json(args.output_dir / "progress.json", progress)
                progress["pairs"].append(pair)
                write_json(args.output_dir / "progress.json", progress)
                print(
                    json.dumps(
                        {
                            "completed": len(progress["pairs"]),
                            "total": len(rows),
                            "clip": row["clip"],
                        }
                    ),
                    flush=True,
                )
            progress["status"] = "PASS"
    except Exception as error:
        progress["status"] = "FAIL"
        progress["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        progress["wall_seconds"] = time.perf_counter() - started
        progress["transferred_bytes"] = {name: r.transferred for name, r in readers.items()}
        write_json(args.output_dir / "progress.json", progress)


if __name__ == "__main__":
    main()
