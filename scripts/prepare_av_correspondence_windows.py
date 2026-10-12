"""Eight fixed training recordings: actual common PTS support, no target payload."""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from scipy.signal import resample_poly

from tiny_omni_decision.audio_interval import sample_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_window(path, cut):
    frames, times = [], []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            if f.pts is None:
                raise ValueError("missing video PTS")
            times.append(Fraction(f.pts) * f.time_base)
            frames.append(f.reformat(width=288, height=288, format="rgb24").to_ndarray())
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("invalid video clock")
    chunks, blocks, rates = [], [], set()
    with av.open(str(path)) as c:
        for f in c.decode(audio=0):
            pcm = f.to_ndarray()
            if not f.format.is_planar or pcm.dtype != np.float32:
                raise ValueError("expected planar float32 decoded PCM")
            chunks.append(pcm.mean(0))
            blocks.append((f.pts, f.time_base, f.samples))
            rates.add(f.sample_rate)
    if len(rates) != 1 or not blocks:
        raise ValueError("missing/changing audio rate")
    rate = rates.pop()
    audio = np.concatenate(chunks)
    start = max(times[0], blocks[0][0] * blocks[0][1], Fraction(0))
    end = min(times[-1], blocks[-1][0] * blocks[-1][1] + Fraction(blocks[-1][2], rate))
    if cut != -1:
        if not 0 <= cut < len(times):
            raise ValueError("cut index cannot be resolved")
        end = min(end, times[cut])
    start_us, end_us = math.ceil(start * 1000000), math.floor(end * 1000000)
    begin, stop = sample_interval(blocks, rate, start_us, end_us)
    eligible = [
        i
        for i, t in enumerate(times)
        if Fraction(start_us, 1000000) <= t < Fraction(end_us, 1000000)
    ]
    if not eligible:
        raise ValueError("empty common video interval")
    selected = [eligible[round(i)] for i in np.linspace(0, len(eligible) - 1, 8)]
    divisor = math.gcd(rate, 24000)
    pcm = resample_poly(audio[begin:stop], 24000 // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(pcm).all():
        raise ValueError("nonfinite audio")
    return dict(audio=pcm[:, None], video=np.stack([frames[i] for i in selected])), dict(
        interval_us=[start_us, end_us],
        sample_bounds=[begin, stop],
        decoded_sample_rate=rate,
        audio_sample_rate=24000,
        selected_frame_indices=selected,
        selected_pts=[str(times[i]) for i in selected],
        cut_frame=cut,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = root / "perception-media-probe-02"
    manifest_path = media / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routing_path = root / "perception-correspondence-plan-01/train-routing.json"
    if sha(routing_path) != "560f3bc7235c842746038e6e5a886471f4dc08cc515d9eba4d12968e608f393b":
        raise ValueError("routing plan changed")
    routing = json.loads(routing_path.read_text(encoding="utf-8"))[:8]
    if [r["audio_video_id"] for r in routing] != [r["video_id"] for r in manifest["clips"]]:
        raise ValueError("pilot media order differs")
    cuts_path = root / "perception-test-source-audit-01/cut_frame_mapping_train.json"
    if sha(cuts_path) != "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4":
        raise ValueError("cut map changed")
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    output = []
    for clip in manifest["clips"]:
        path = media / clip["file"]
        if sha(path) != clip["sha256"] or clip["metadata"]["split"] != "train":
            raise ValueError("train media changed")
        cut = cuts[clip["video_id"]]
        payload, timing = decode_window(path, cut)
        target = args.output / f"window-{len(output):03}.npz"
        np.savez_compressed(target, **payload)
        output.append(
            dict(
                id=clip["video_id"],
                status="prepared",
                path=str(target),
                sha256=sha(target),
                media_sha256=clip["sha256"],
                metadata_sample_rate=clip["metadata"]["audio_sample_rate"],
                **timing,
            )
        )
    report = dict(
        windows=output,
        manifest_sha256=sha(manifest_path),
        routing_sha256=sha(routing_path),
        cuts_sha256=sha(cuts_path),
        training=False,
        evaluation=False,
        scope="Common decoded interval; last video PTS excluded conservatively",
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                windows=len(output),
                rate_mismatches=[
                    r["id"] for r in output if r["decoded_sample_rate"] != r["metadata_sample_rate"]
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
