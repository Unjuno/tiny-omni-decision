"""Offline CPU decode and raw annotation accounting of the fixed train-only probe."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts.greatest_hits_source import parse_ids, parse_times
from scripts.probe_greatest_hits_train import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only process required")
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    # Optional decoder packages are needed only by this explicit physical probe.
    import av
    import numpy as np

    receipt = json.loads((args.probe / "result.json").read_text())
    if receipt["status"] != "PASS_FIXED_TRAIN_MEMBER_ACQUISITION":
        raise ValueError("successful bounded acquisition receipt required")
    for name, evidence in receipt["fetched"].items():
        if Path(name).name != name or sha(args.probe / name) != evidence["sha256"]:
            raise ValueError("source member changed")
    selected = receipt["selected"]
    media = args.probe / (selected[0] + "_denoised.mp4")
    train, test = [parse_ids((args.probe / (s + ".txt")).read_bytes()) for s in ("train", "test")]
    if not set(selected) <= set(train) or set(train) & set(test):
        raise ValueError("selected IDs must be official train only")
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(status="RUNNING", command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  probe_receipt_sha256=sha(args.probe / "result.json"),
                  python=platform.python_version(), pyav=av.__version__, numpy=np.__version__,
                  ffmpeg_libraries={k: list(v) for k, v in av.library_versions.items()},
                  streams=[], annotations={}, test_media_or_labels_opened=False,
                  model_loaded=False, optimizer_updates=0, sealed_audit_opened=False)
    try:
        all_rows = []
        for ident in selected:
            rows = parse_times((args.probe / (ident + "_times.txt")).read_bytes())
            all_rows.extend(rows)
            result["annotations"][ident] = dict(
                count=len(rows), complete=sum(None not in r["fields"] for r in rows),
                all_missing=sum(all(x is None for x in r["fields"]) for r in rows), rows=rows)
        result["raw_label_field_counts"] = [dict(Counter(
            "<MISSING>" if r["fields"][i] is None else r["fields"][i] for r in all_rows))
            for i in range(3)]
        # These prefixes are conservative grouping diagnostics, not proven session IDs.
        for label, components in [("timestamp_prefix", 6), ("date_prefix", 3)]:
            groups = [{"-".join(i.split("-")[:components]) for i in ids} for ids in [train, test]]
            result[label] = dict(train=len(groups[0]), test=len(groups[1]),
                                 shared=len(groups[0] & groups[1]), identity_semantics="UNKNOWN")
        with av.open(str(media)) as container:
            stream_ids = [(s.index, s.type) for s in container.streams]
            duration = container.duration / av.time_base
        if Counter(t for _, t in stream_ids) != Counter({"video": 1, "audio": 1}):
            raise ValueError("expected exactly one real video and one audio stream")
        thumbs = []
        for index, kind in stream_ids:
            pts, ends, fingerprints = [], [], set()
            finite, samples, energy, peak, next_thumb = True, 0, 0.0, 0.0, 0
            with av.open(str(media)) as container:
                stream = container.streams[index]
                stream.thread_type = "NONE"
                info = dict(index=index, type=kind, codec=stream.codec_context.name,
                            time_base=str(stream.time_base), duration=float(stream.duration
                                * stream.time_base), metadata=dict(stream.metadata))
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("decoded frame lacks PTS")
                    stamp = float(frame.pts * frame.time_base)
                    if pts and stamp <= pts[-1]:
                        raise ValueError("nonmonotonic PTS")
                    pts.append(stamp)
                    if kind == "audio":
                        data = frame.to_ndarray()
                        finite = finite and bool(np.isfinite(data).all())
                        if not np.issubdtype(data.dtype, np.floating):
                            raise ValueError("audio scaling needs explicit review")
                        samples += frame.samples
                        energy += float(np.sum(data.astype(np.float64) ** 2))
                        peak = max(peak, float(np.max(np.abs(data))))
                        ends.append(stamp + frame.samples / frame.sample_rate)
                        info.update(sample_rate=frame.sample_rate,
                                    channels=len(frame.layout.channels),
                                    decoded_dtype=str(data.dtype))
                    else:
                        data = frame.to_ndarray(format="rgb24")
                        fingerprints.add(hashlib.sha256(data.tobytes()).hexdigest())
                        ends.append(stamp + float(1 / stream.average_rate))
                        info.update(width=frame.width, height=frame.height,
                                    average_rate=str(stream.average_rate))
                        if next_thumb < 5 and stamp >= duration * next_thumb / 5:
                            thumbs.append((stamp, data))
                            next_thumb += 1
            if not pts or not finite:
                raise ValueError("empty or nonfinite stream")
            info.update(decoded_frames=len(pts), first_pts=pts[0], last_pts=pts[-1],
                        decoded_end=ends[-1], pts_strictly_increasing=True)
            if kind == "audio":
                if energy <= 0:
                    raise ValueError("silent audio stream")
                info.update(samples_per_channel=samples, all_finite=finite, peak=peak,
                            rms=(energy / (samples * info["channels"])) ** 0.5)
            else:
                info["distinct_decoded_frames"] = len(fingerprints)
            result["streams"].append(info)
            (args.output / (kind + "-timestamps.json")).write_text(
                json.dumps(dict(pts=pts, ends=ends)), encoding="utf-8")
        result["raw_frame_previews"] = []
        for i, (stamp, pixels) in enumerate(thumbs):
            name = f"raw-train-frame-{i:02d}.png"
            with av.open(str(args.output / name), mode="w", format="image2") as output:
                stream = output.add_stream("png", rate=1)
                stream.width, stream.height = pixels.shape[1], pixels.shape[0]
                stream.pix_fmt = "rgb24"
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            result["raw_frame_previews"].append(dict(file=name, pts=stamp,
                                                       sha256=sha(args.output / name)))
        first_rows = result["annotations"][selected[0]]["rows"]
        result["first_parent_annotation_times_within_both_streams"] = all(
            s["first_pts"] <= r["seconds"] < s["decoded_end"]
            for r in first_rows for s in result["streams"])
        result["physical_av_alignment"] = "UNKNOWN; matching container PTS is insufficient"
        for name, evidence in receipt["fetched"].items():
            if sha(args.probe / name) != evidence["sha256"]:
                raise ValueError("source changed during decode")
        result["status"] = "PASS_DECODE_AND_RAW_LABEL_ACCOUNTING"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        result.update(seconds=time.monotonic() - started, utc=datetime.now(UTC).isoformat(),
                      source_hashes={p: sha(repo / p) for p in [
                          "scripts/greatest_hits_source.py",
                          "scripts/inspect_greatest_hits_train.py"]})
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "annotations"}))


if __name__ == "__main__":
    main()
