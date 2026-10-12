"""Pinned complete-pair dense video features; no fitting or quality scoring."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import av
import numpy as np

from scripts.extract_starss23_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    load_encoder,
    sha256_file,
)
from scripts.prepare_av_correspondence_windows import decode_window
from tiny_omni_decision.temporal_sampling_plan import plan_windows

PINS = {
    "train": "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",
    "valid": "213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546",
}
PAIR_PINS = {
    "train": "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7",
    "valid": "30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2",
}


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    args.modality = "video"
    args.source_root = r1 / "videoprism-base-r1/source"
    args.checkpoint = r1 / "videoprism-base-f16r288.npz"
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint changed")
    if (
        subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("encoder source changed")
    clips = []
    for split, pin in PINS.items():
        manifest = root / f"perception-occlusion-{split}-media-01/manifest.json"
        pairs = root / f"perception-occlusion-pairs-01/{split}.json"
        if sha256_file(manifest) != pin or sha256_file(pairs) != PAIR_PINS[split]:
            raise ValueError("input manifest changed")
        rows = json.loads(manifest.read_text())["clips"]
        ids = [r["video_id"] for r in json.loads(pairs.read_text())["pairs"]]
        if ids != [r["video_id"] for r in rows]:
            raise ValueError("complete-pair order mismatch")
        clips.extend(dict(split=split, **r) for r in rows)
    if len({r["video_id"] for r in clips}) != 356:
        raise ValueError("identity overlap/count mismatch")
    args.output.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        training=False,
        quality_evaluation=False,
        started_unix=time.time(),
        manifest_hashes=PINS,
        pair_hashes=PAIR_PINS,
        checkpoint_sha256=CHECKPOINT_SHA256,
        encoder_source_revision=SOURCE_COMMIT,
        script_sha256=sha256_file(Path(__file__)),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        fps=4,
        width=8,
        device="cpu",
        features=[],
        interval_policy="unchanged common audio/video PTS support, exclusive reveal cut",
    )
    hashes = {"train": set(), "valid": set()}
    window_hashes = {"train": set(), "valid": set()}
    try:
        encode, _ = load_encoder(args, report)
        report["preprocessing"] = "PyAV RGB resize 288x288; float32 /255; 4fps 8-frame windows"
        for row in clips:
            tick = time.perf_counter()
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError("media changed")
            old, timing = decode_window(path, row["cut_frame"])
            del old
            stamps = []
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        raise ValueError("missing PTS")
                    stamps.append(float(frame.pts * frame.time_base))
            start, end = [x / 1e6 for x in timing["interval_us"]]
            plan = plan_windows(stamps, start=start, end=end)
            frames = {}
            wanted = set(plan["indices"])
            with av.open(str(path)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i in wanted:
                        frames[i] = frame.reformat(
                            width=288, height=288, format="rgb24"
                        ).to_ndarray()
                    if i >= plan["indices"][-1]:
                        break
            digest = array_hash(np.stack([frames[i] for i in plan["indices"]]))
            other = "valid" if row["split"] == "train" else "train"
            if digest in hashes[other]:
                raise ValueError("cross-split prepared-video overlap")
            hashes[row["split"]].add(digest)
            features, inputs, seconds = [], [], []
            for indices in plan["windows"]:
                pixels = np.stack([frames[i] for i in indices])
                key = array_hash(pixels)
                if key in window_hashes[other]:
                    raise ValueError("cross-split prepared-window overlap")
                window_hashes[row["split"]].add(key)
                inputs.append(key)
                begin = time.perf_counter()
                feature = encode({"video": pixels})
                seconds.append(time.perf_counter() - begin)
                if feature.shape != (768,) or not np.isfinite(feature).all():
                    raise ValueError("invalid features")
                features.append(feature)
            target = args.output / f"{row['split']}-{row['video_id']}.npy"
            features = np.stack(features)
            np.save(target, features, allow_pickle=False)
            np.testing.assert_array_equal(features, np.load(target, allow_pickle=False))
            report["features"].append(
                dict(
                    video_id=row["video_id"],
                    split=row["split"],
                    media_sha256=row["sha256"],
                    path=str(target),
                    sha256=sha256_file(target),
                    shape=list(features.shape),
                    prepared_sha256=digest,
                    window_input_hashes=inputs,
                    timing=timing,
                    plan=plan,
                    window_seconds=seconds,
                    seconds=time.perf_counter() - tick,
                )
            )
            (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                f"{len(report['features'])}/356 {row['video_id']} windows={len(features)}",
                flush=True,
            )
        report.update(status="PASS_FEATURE_EXTRACTION_NOT_QUALITY", prepared_cross_split_overlap=0)
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["ended_unix"] = time.time()
        (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
