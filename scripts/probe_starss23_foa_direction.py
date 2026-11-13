"""Training-only fixed broadband direction diagnostic; no model fitting."""

import csv
import io
import json
import time
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.extract_frozen_videoprism_features import sha256_file
from tiny_omni_decision.foa_direction import foa_direction


def summary(rows):
    errors = np.array([r["error_deg"] for r in rows])
    return dict(count=len(rows), mean_deg=float(errors.mean()), median_deg=float(np.median(errors)),
                within20=float((errors <= 20).mean()), within45=float((errors <= 45).mean()))


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split = root / "starss23-room-splits-v1/train.jsonl"
    media_manifest = root / "starss23-all-windows-v1/train.jsonl"
    metadata = root / "starss23-source-audit-01/metadata_dev.zip"
    pins = {split: "571c2b537a809b2f92c9289fd932c20e0169ca1c0e38f4dc0f4c6ad6ef2ac978",
            media_manifest: "2d06c0e94ea68d9b5372eb942f37cd87fcf4d9ba71f05ae71bb3be836ae73347",
            metadata: "2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6"}
    for path, digest in pins.items():
        if sha256_file(path) != digest:
            raise ValueError("input hash mismatch")
    records = [json.loads(line) for line in split.read_text().splitlines()]
    if len(records) != 49 or any(r["split"] != "train" for r in records):
        raise ValueError("fixed training scope mismatch")
    media = {}
    for line in media_manifest.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != "train":
            raise ValueError("training media only")
        previous = media.setdefault(row["recording"], row["media"]["audio"])
        if previous != row["media"]["audio"]:
            raise ValueError("conflicting audio")
    out = root / "starss23-foa-direction-probe-01"
    out.mkdir(exist_ok=False)
    config = dict(input_hashes={str(k): v for k, v in pins.items()},
                  script_sha256=sha256_file(Path(__file__)),
                  feature_sha256=sha256_file(Path("src/tiny_omni_decision/foa_direction.py")),
                  policy="all single-annotation 100ms ticks; no filtering or fit; W,Y,Z,X",
                  scoring="angular distance to annotated unit vector; report excluded tick counts",
                  inference_available="after 100ms block and calculation complete")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    results, counts = [], Counter()
    start = time.perf_counter()
    with zipfile.ZipFile(metadata) as archive, (out / "predictions.jsonl").open("w") as saved:
        for record in records:
            info = media[record["clip"]]
            if not info["member"].startswith("foa_dev/"):
                raise ValueError("FOA media required")
            if sha256_file(Path(info["path"])) != info["sha256"]:
                raise ValueError("audio changed")
            with wave.open(info["path"]) as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 4, 2):
                    raise ValueError("audio format mismatch")
                samples = np.frombuffer(wav.readframes(wav.getnframes()),
                                        dtype="<i2").reshape(-1, 4)
            labels = defaultdict(list)
            for row in csv.reader(io.StringIO(archive.read(record["member"]).decode())):
                tick, label, source, azimuth, elevation, distance = map(int, row)
                labels[tick].append((label, source, azimuth, elevation, distance))
            counts["complete_audio_ticks"] += len(samples) // 2400
            counts["annotated_ticks"] += len(labels)
            for tick, entries in sorted(labels.items()):
                if len(entries) != 1:
                    counts["multiple_annotation_ticks"] += 1
                    continue
                block = samples[tick * 2400:(tick + 1) * 2400]
                if len(block) != 2400:
                    counts["incomplete_audio_ticks"] += 1
                    continue
                vector = foa_direction(block.astype(np.float64) / 32768)
                if vector is None:
                    counts["zero_direction_ticks"] += 1
                    continue
                label, source, az, el, distance = entries[0]
                a, e = np.deg2rad([az, el])
                truth = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
                error = float(np.rad2deg(np.arccos(np.clip(vector @ truth, -1, 1))))
                entry = dict(recording=record["clip"], room=record["room"], tick=tick,
                             raw_class=label, source=source, distance_cm=distance,
                             predicted_xyz=vector.tolist(), target_xyz=truth.tolist(),
                             error_deg=error)
                results.append(entry)
                saved.write(json.dumps(entry) + "\n")
            print(record["clip"], flush=True)
    if not results:
        raise ValueError("no diagnostic rows")
    report = dict(summary=summary(results), counts=dict(counts),
                  per_room={room: summary([r for r in results if r["room"] == room])
                            for room in sorted({r["room"] for r in results})},
                  seconds=time.perf_counter() - start,
                  predictions_sha256=sha256_file(out / "predictions.jsonl"),
                  validation_read=False, sealed_audit_read=False, parameters_trained=0,
                  limitation="single annotated event does not guarantee single acoustic source")
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
