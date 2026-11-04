"""Extract a complete frozen STARSS23 window manifest using one model load."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import groupby
from pathlib import Path

from scripts.extract_frozen_videoprism_features import CHECKPOINT_SHA256, SOURCE_COMMIT, sha256_file
from scripts.fetch_starss23_frozen_media import write_json
from scripts.starss23_stream import audio_windows, video_windows


def read_manifest(root):
    result = json.loads((root / "result.json").read_text())
    if result.get("status") != "PASS":
        raise ValueError("incomplete window manifest")
    rows = []
    for split in ("train", "calibration", "validation"):
        path = root / f"{split}.jsonl"
        if sha256_file(path) != result["splits"][split]["sha256"]:
            raise ValueError("window manifest hash mismatch")
        part = [json.loads(line) for line in path.read_text().splitlines()]
        if len(part) != result["splits"][split]["windows"] or any(
            r["split"] != split for r in part
        ):
            raise ValueError("window split/count mismatch")
        rows.extend(part)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate window IDs")
    seen = {}
    for row in rows:
        for key in ("recording", "room"):
            previous = seen.setdefault((key, row[key]), row["split"])
            if previous != row["split"]:
                raise ValueError("cross-split group overlap")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["audio", "video"], required=True)
    for key in ("manifest-dir", "source-root", "checkpoint", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest_dir)
    import numpy as np
    import psutil

    from scripts.extract_starss23_features import AUDIO_SHA, AUDIO_SOURCE, load_encoder

    expected_sha = AUDIO_SHA if args.modality == "audio" else CHECKPOINT_SHA256
    revision = AUDIO_SOURCE if args.modality == "audio" else SOURCE_COMMIT
    if sha256_file(args.checkpoint) != expected_sha:
        raise ValueError("checkpoint hash mismatch")
    if (
        subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != revision
    ):
        raise ValueError("model source revision mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    process = psutil.Process()
    report = {
        "status": "RUNNING",
        "modality": args.modality,
        "checkpoint_sha256": expected_sha,
        "source_commit": revision,
        "manifest_sha256": sha256_file(args.manifest_dir / "result.json"),
        "runner_sha256": sha256_file(Path(__file__)),
        "features": [],
        "recording_shards": [],
        "device": "cpu",
        "training": False,
    }
    try:
        encode, shape = load_encoder(args, report)
        report["load_seconds"] = time.perf_counter() - started
        vectors = []
        for recording, group in groupby(rows, key=lambda r: r["recording"]):
            group = list(group)
            first_index = len(vectors)
            media = group[0]["media"][args.modality]
            if any(r["media"][args.modality] != media for r in group):
                raise ValueError("recording maps to inconsistent media")
            if sha256_file(Path(media["path"])) != media["sha256"]:
                raise ValueError("media hash mismatch")
            windows = [(r["start_seconds"], r["end_seconds"]) for r in group]
            iterator = (audio_windows if args.modality == "audio" else video_windows)(
                media["path"], windows
            )
            for row, data in zip(group, iterator, strict=True):
                tick = time.perf_counter()
                feature = encode({args.modality: data if args.modality == "audio" else data[0]})
                if feature.shape != shape or not np.isfinite(feature).all():
                    raise ValueError("invalid feature")
                vectors.append(feature)
                report["features"].append(
                    {
                        "id": row["id"],
                        "recording": recording,
                        "split": row["split"],
                        "media_sha256": media["sha256"],
                        "seconds": time.perf_counter() - tick,
                    }
                )
            shard_path = args.output_dir / f"recording-{len(report['recording_shards']):03d}.npy"
            shard = np.stack(vectors[first_index:]).astype(np.float32)
            np.save(shard_path, shard, allow_pickle=False)
            if not np.array_equal(shard, np.load(shard_path, allow_pickle=False)):
                raise ValueError("recording shard reload mismatch")
            report["recording_shards"].append(
                {
                    "recording": recording,
                    "first_index": first_index,
                    "count": len(group),
                    "path": str(shard_path.resolve()),
                    "sha256": sha256_file(shard_path),
                }
            )
            write_json(args.output_dir / "progress.json", report)
            print(
                json.dumps({"recording": recording, "features": len(vectors), "total": len(rows)}),
                flush=True,
            )
        features = np.stack(vectors).astype(np.float32)
        path = args.output_dir / "features.npy"
        np.save(path, features, allow_pickle=False)
        if not np.array_equal(features, np.load(path, allow_pickle=False)):
            raise ValueError("feature reload mismatch")
        report.update(
            status="PASS", feature_shape=list(features.shape), feature_sha256=sha256_file(path)
        )
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["peak_working_set_bytes"] = getattr(process.memory_info(), "peak_wset", None)
        write_json(args.output_dir / "result.json", report)


if __name__ == "__main__":
    main()
