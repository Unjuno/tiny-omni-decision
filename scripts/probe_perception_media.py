"""Two label-independent train clips, bounded archive reads and stream timing."""

import argparse
import hashlib
import json
import subprocess
import zipfile
import zlib
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(folder):
    import av

    if (folder / "timing.json").exists():
        raise ValueError("timing report already exists")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for record in manifest["clips"]:
        path = folder / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError("media changed")
        streams = {}
        for kind in ("video", "audio"):
            times, ends, samples = [], [], 0
            with av.open(str(path)) as container:
                selected = getattr(container.streams, kind)
                if len(selected) != 1:
                    raise ValueError("expected one stream of each modality")
                stream = selected[0]
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("missing presentation timestamp")
                    t = float(frame.pts * frame.time_base)
                    times.append(t)
                    if kind == "audio":
                        samples += frame.samples
                        ends.append(t + frame.samples / frame.sample_rate)
                    else:
                        ends.append(t + 1 / float(stream.average_rate))
            if not times or any(b <= a for a, b in zip(times, times[1:], strict=False)):
                raise ValueError("empty or nonmonotonic stream")
            streams[kind] = dict(
                frames=len(times),
                start_seconds=times[0],
                end_seconds=ends[-1],
                decoded_audio_samples=samples,
            )
        results.append(
            dict(
                video_id=record["video_id"],
                streams=streams,
                annotation_metadata=record["metadata"],
                av_end_delta_seconds=streams["audio"]["end_seconds"]
                - streams["video"]["end_seconds"],
            )
        )
    (folder / "timing.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--count", type=int, choices=(2, 8), default=2)
    args = parser.parse_args()
    if args.decode:
        decode(args.output)
        return
    from scripts.audit_mmfit import RemoteZip

    source = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip"
    )
    if digest(source) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("annotation revision mismatch")
    args.output.mkdir(exist_ok=False)
    with zipfile.ZipFile(source) as z:
        annotations = json.loads(z.read("all_train.json"))
    ids = sorted(annotations, key=lambda x: hashlib.sha256(f"17:{x}".encode()).hexdigest())[
        : args.count
    ]
    url = "https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip?generation=1685446268139925"
    remote = RemoteZip(url, 28404087974, '"-CJXz5tD4nP8CEAE="')
    records = []
    with zipfile.ZipFile(remote) as z:
        for video_id in ids:
            matches = [i for i in z.infolist() if Path(i.filename).name == video_id + ".mp4"]
            if len(matches) != 1 or matches[0].file_size > 32_000_000:
                raise ValueError("missing/duplicate/oversized selected clip; do not substitute")
            entry = matches[0]
            path = args.output / (video_id + ".mp4")
            prior = Path("C:/CodexArtifacts/pretrained-reuse-r2/perception-media-probe-01")
            prior_manifest = (
                json.loads((prior / "manifest.json").read_text(encoding="utf-8"))
                if (prior / "manifest.json").exists()
                else {"clips": []}
            )
            cached = next((r for r in prior_manifest["clips"] if r["video_id"] == video_id), None)
            if cached:
                original = prior / cached["file"]
                if (
                    digest(original) != cached["sha256"]
                    or zlib.crc32(original.read_bytes()) != entry.CRC
                ):
                    raise ValueError("existing clip changed")
                with path.open("xb") as dst:
                    dst.write(original.read_bytes())
            else:
                with z.open(entry) as src, path.open("xb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
            records.append(
                dict(
                    video_id=video_id,
                    file=path.name,
                    sha256=digest(path),
                    bytes=path.stat().st_size,
                    crc32=entry.CRC,
                    metadata=annotations[video_id]["metadata"],
                )
            )
    manifest = dict(
        url=url,
        archive_bytes=28404087974,
        selection=f"sha256(17:video_id), first {args.count}",
        source_sha256=digest(source),
        clips=records,
        training=False,
        evaluation=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    python = "C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe"
    subprocess.run(
        [python, "-m", "scripts.probe_perception_media", "--decode", "--output", str(args.output)],
        check=True,
    )


if __name__ == "__main__":
    main()
