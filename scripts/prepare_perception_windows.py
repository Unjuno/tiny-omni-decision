"""PTS-selected train pilot windows; no supervision in encoder payloads."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from scipy.signal import resample_poly

from tiny_omni_decision.audio_interval import sample_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = root / "perception-media-probe-01"
    manifest = json.loads((media / "manifest.json").read_text(encoding="utf-8"))
    proposal = root / "perception-visibility-proposal-01/records.jsonl"
    if sha(proposal) != "f2d19f6c148bd44c14d1f9b6949bf129f0e3fd2df300dedac0257a55cef7ac4a":
        raise ValueError("proposal changed")
    rows = [json.loads(line) for line in proposal.read_text(encoding="utf-8").splitlines()]
    output = []
    for clip in manifest["clips"]:
        path = media / clip["file"]
        if sha(path) != clip["sha256"]:
            raise ValueError("media changed")
        frames, timestamps = [], []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                timestamps.append(Fraction(frame.pts) * frame.time_base)
                frames.append(frame.reformat(width=288, height=288, format="rgb24").to_ndarray())
        if any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("nonmonotonic video PTS")
        chunks, blocks = [], []
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                if frame.sample_rate != 48000 or not frame.format.is_planar:
                    raise ValueError("pilot requires planar 48 kHz decoded audio")
                chunks.append(frame.to_ndarray().mean(0))
                blocks.append((frame.pts, frame.time_base, frame.samples))
        audio = np.concatenate(chunks)
        if audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise ValueError("finite float32 decoded PCM required")
        for row in rows:
            if row["group_id"] != clip["video_id"]:
                continue
            record = dict(id=row["record_id"], proposal_status=row["status"])
            if row["status"] != "candidate":
                record.update(status="held", reason=row["hold_reasons"])
                output.append(record)
                continue
            start, end = row["request"]["interval_us"]
            indices = [
                i
                for i, t in enumerate(timestamps)
                if Fraction(start, 1000000) <= t < Fraction(end, 1000000)
            ]
            if not indices:
                record.update(status="held", reason="no_video_frame_in_interval")
                output.append(record)
                continue
            selected = [indices[round(x)] for x in np.linspace(0, len(indices) - 1, 8)]
            begin, stop = sample_interval(blocks, 48000, start, end)
            pcm = resample_poly(audio[begin:stop], 1, 2).astype(np.float32)[:, None]
            target = args.output / f"window-{len(output):03}.npz"
            np.savez_compressed(target, audio=pcm, video=np.stack([frames[i] for i in selected]))
            record.update(
                status="prepared",
                path=str(target),
                sha256=sha(target),
                interval_us=[start, end],
                sample_bounds_48khz=[begin, stop],
                selected_frame_indices=selected,
                selected_pts_seconds=[str(timestamps[i]) for i in selected],
                unique_frames=len(set(selected)),
                audio_sample_rate=24000,
            )
            output.append(record)
    report = dict(
        windows=output,
        proposal_sha256=sha(proposal),
        preprocessing="PyAV 288x288 RGB reformat; channel mean; scipy 48->24k resample_poly",
        training=False,
        evaluation=False,
        interpretation="Technical train pilot only",
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(dict(total=len(output), prepared=sum(r["status"] == "prepared" for r in output)))
    )


if __name__ == "__main__":
    main()
