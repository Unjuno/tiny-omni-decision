"""Separate CPU resampler contract or fixed three-TRAIN-parent physical input probe."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from scripts.audit_greatest_hits_development_media import ANNOTATIONS_SHA
from scripts.fetch_greatest_hits_development_media import PARTITION_SHA
from scripts.fetch_greatest_hits_train_inventory import write
from scripts.greatest_hits_inputs import crop_audio, resample_direct, select_video
from scripts.probe_greatest_hits_train import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("resampler", "physical"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only process required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ValueError("clean committed source required")
    import av
    import numpy as np

    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(status="RUNNING", phase=args.phase, command=[sys.executable, *sys.argv],
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  python=platform.python_version(), pyav=av.__version__, numpy=np.__version__,
                  ffmpeg_libraries={k: list(v) for k, v in av.library_versions.items()},
                  model_loaded=False, optimizer_updates=0, validation_loaded=False,
                  sealed_audit_loaded=False, rows=[], artifacts={})

    def save(name, value):
        path = args.output / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        if not np.array_equal(np.load(path, allow_pickle=False), value):
            raise ValueError("input save/reload mismatch")
        result["artifacts"][path.name] = sha(path)

    try:
        if args.phase == "resampler":
            t = np.arange(48000, dtype=np.float64) / 96000
            for frequency in (4000, 12000):
                native = (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
                values = {rate: resample_direct(native, rate) for rate in (16000, 32000)}
                rms = {str(rate): float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))
                       for rate, v in values.items()}
                if ((frequency == 4000 and not all(0.15 < v < 0.20 for v in rms.values()))
                        or (frequency == 12000 and not (rms["16000"] < 0.01
                                                       and 0.15 < rms["32000"] < 0.20))):
                    raise ValueError("direct resampler frequency response differs")
                for rate, value in values.items():
                    if not np.array_equal(value, resample_direct(native, rate)):
                        raise ValueError("resampling repeat differs")
                    save(f"tone-{frequency}-{rate}", value)
                save(f"tone-{frequency}-96000", native)
                result["rows"].append(dict(tone_hz=frequency, amplitude=0.25, output_rms=rms))
        else:
            root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
            inventory = root / "greatest-hits-train-inventory-01"
            media = root / "greatest-hits-development-media-01"
            audit = root / "greatest-hits-development-decode-01"
            if (sha(inventory / "annotations.json") != ANNOTATIONS_SHA
                    or sha(inventory / "partition.json") != PARTITION_SHA):
                raise ValueError("frozen inputs changed")
            completed = json.loads((audit / "result.json").read_text())
            if completed["status"] != "PASS_COMPLETE_DECODE":
                raise ValueError("complete media audit required")
            plan = json.loads((inventory / "partition.json").read_text())
            annotations = json.loads((inventory / "annotations.json").read_text())
            members = json.loads((media / "members.json").read_text())
            selected = sorted(plan["train"], key=lambda i: hashlib.sha256(
                ("17:" + i).encode()).hexdigest())[:3]
            if selected != ["2015-03-30-00-26-27", "2015-03-12-20-15-29", "2015-03-31-02-08-23"]:
                raise ValueError("fixed TRAIN selection changed")
            for ident in selected:
                path = media / "members" / (ident + "_denoised.mp4")
                if sha(path) != members[ident]["sha256"]:
                    raise ValueError("source media changed")
                pieces, first, previous_end = [], None, None
                with av.open(str(path)) as container:
                    for frame in container.decode(audio=0):
                        stamp = float(frame.pts * frame.time_base)
                        if frame.sample_rate != 96000 or len(frame.layout.channels) != 1:
                            raise ValueError("native mono96kHz required")
                        if previous_end is not None and abs(stamp - previous_end) > 1e-8:
                            raise ValueError("PCM timestamps discontinuous")
                        first = stamp if first is None else first
                        previous_end = stamp + frame.samples / frame.sample_rate
                        pieces.append(frame.to_ndarray().reshape(-1))
                native = np.concatenate(pieces)
                onset = annotations[ident][0]["seconds"]
                cropped, timing = crop_audio(native, 96000, first, onset)
                with av.open(str(path)) as container:
                    pts = np.array([float(f.pts * f.time_base) for f in container.decode(video=0)])
                indices = select_video(pts, timing["start_seconds"], timing["end_seconds"])
                frames = []
                with av.open(str(path)) as container:
                    for index, frame in enumerate(container.decode(video=0)):
                        if index in indices:
                            frames.append(frame.to_ndarray(format="rgb24"))
                        if index >= indices[-1]:
                            break
                rgb = np.stack(frames)
                if len(rgb) != 8:
                    raise ValueError("exact eight source frames required")
                contact = np.concatenate([np.concatenate(list(rgb[:4]), axis=1),
                                          np.concatenate(list(rgb[4:]), axis=1)], axis=0)
                preview = args.output / (ident + "-contact.png")
                with av.open(str(preview), mode="w", format="image2") as output:
                    stream = output.add_stream("png", rate=1)
                    stream.width, stream.height = contact.shape[1], contact.shape[0]
                    stream.pix_fmt = "rgb24"
                    for packet in stream.encode(av.VideoFrame.from_ndarray(contact, format="rgb24")):
                        output.mux(packet)
                    for packet in stream.encode():
                        output.mux(packet)
                result["artifacts"][preview.name] = sha(preview)
                save(ident + "-rgb", rgb)
                save(ident + "-native-96000", cropped)
                for rate in (16000, 32000):
                    converted = resample_direct(cropped, rate)
                    if not np.array_equal(converted, resample_direct(cropped, rate)):
                        raise ValueError("physical resampling repeat differs")
                    save(ident + f"-direct-{rate}", converted)
                save(ident + "-video-pts", pts[indices])
                if sha(path) != members[ident]["sha256"]:
                    raise ValueError("source changed during input preparation")
                result["rows"].append(dict(parent=ident, event_row=0, source_sha256=sha(path),
                    onset_seconds=onset, audio=timing, video_pts=pts[indices].tolist(),
                    rgb_shape=list(rgb.shape), native_audio_peak=float(np.max(np.abs(cropped)))))
        result["status"] = "PASS_FIXED_INPUT_PROBE"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        result.update(seconds=time.monotonic() - started,
                      source_hashes={p: sha(repo / p) for p in [
                          "scripts/greatest_hits_inputs.py",
                          "scripts/probe_greatest_hits_inputs.py"]})
        write(args.output / "result.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
