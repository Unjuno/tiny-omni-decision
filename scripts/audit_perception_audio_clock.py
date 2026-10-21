"""Inspect actual packets and apply exact interval plans to the two pinned clips."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import av

from tiny_omni_decision.audio_interval import sample_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = root / "perception-media-probe-01"
    manifest = json.loads((media / "manifest.json").read_text(encoding="utf-8"))
    archive = root / "perception-test-source-audit-01/train_annotations.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("annotation bytes changed")
    with zipfile.ZipFile(archive) as z:
        annotations = json.loads(z.read("all_train.json"))
    records = []
    for clip in manifest["clips"]:
        path = media / clip["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError("media changed")
        with av.open(str(path)) as c:
            stream = c.streams.audio[0]
            packets = [p for p in c.demux(stream) if p.size]
            packet_info = dict(
                first_pts=packets[0].pts,
                time_base=str(stream.time_base),
                stream_duration=stream.duration,
                packet_duration_total=sum(p.duration for p in packets),
                first_side_data=[
                    dict(type=str(s.data_type), hex=bytes(s).hex())
                    for s in packets[0].iter_sidedata()
                ],
            )
        with av.open(str(path)) as c:
            rate = c.streams.audio[0].codec_context.sample_rate
            blocks = [(f.pts, f.time_base, f.samples) for f in c.decode(audio=0)]
        plans = []
        for sound in annotations[clip["video_id"]]["sound_localisation"]:
            try:
                bounds = sample_interval(blocks, rate, *sound["timestamps"])
                plans.append(dict(id=sound["id"], status="covered", sample_bounds=bounds))
            except ValueError as e:
                plans.append(dict(id=sound["id"], status="held", reason=str(e)))
        records.append(
            dict(
                video_id=clip["video_id"],
                packets=packet_info,
                decoded_samples=sum(b[2] for b in blocks),
                annotation_samples=clip["metadata"]["audio_samples"],
                plans=plans,
            )
        )
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                dict(
                    video_id=r["video_id"],
                    decoded_samples=r["decoded_samples"],
                    covered=sum(p["status"] == "covered" for p in r["plans"]),
                    held=sum(p["status"] == "held" for p in r["plans"]),
                )
                for r in records
            ]
        )
    )


if __name__ == "__main__":
    main()
