"""One fixed train recording: dense window cost, not decision quality."""

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    plan_path = root / "perception-dense-temporal-plan-01.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    row = plan["videos"][0]
    if row["id"] != "video_3334" or plan["fps"] != 4 or plan["width"] != 8:
        raise ValueError("fixed smoke changed")
    media = root / "perception-correspondence-train-media-01/video_3334.mp4"
    if sha256_file(media) != "e511a81e7d1f1b5f533c3be134e6e3604a4789f52eb807903405476c2bff996d":
        raise ValueError("media changed")
    args.modality = "video"
    args.source_root = r1 / "videoprism-base-r1/source"
    args.checkpoint = r1 / "videoprism-base-f16r288.npz"
    if (
        sha256_file(args.checkpoint) != CHECKPOINT_SHA256
        or subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("encoder changed")
    args.output.mkdir(exist_ok=False)
    report = dict(
        plan_sha256=sha256_file(plan_path),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha256_file(Path(__file__)),
        training=False,
        quality_evaluation=False,
    )
    start = time.perf_counter()
    frames = {}
    with av.open(str(media)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= row["cut_frame"]:
                break
            if i in row["indices"]:
                frames[i] = frame.reformat(width=288, height=288, format="rgb24").to_ndarray()
    report["decode_seconds"] = time.perf_counter() - start
    encode, _ = load_encoder(args, report)
    report["preprocessing"] = "PyAV RGB resize 288x288; /255; fixed PTS windows"
    old_path = root / "perception-correspondence-development-windows-01/window-016.npz"
    old_manifest = json.loads((old_path.parent / "result.json").read_text(encoding="utf-8"))
    if sha256_file(old_path) != old_manifest["windows"][16]["sha256"]:
        raise ValueError("old window changed")
    with np.load(old_path, allow_pickle=False) as data:
        old = encode(data)
        start = time.perf_counter()
        repeated = encode(data)
        report["old_warm_seconds"] = time.perf_counter() - start
    np.testing.assert_array_equal(old, repeated)
    reference = root / "perception-correspondence-development-video-01/feature-016.npy"
    np.testing.assert_allclose(old, np.load(reference), atol=1e-6, rtol=1e-5)
    features, seconds = [], []
    for indices in row["windows"]:
        start = time.perf_counter()
        feature = encode({"video": np.stack([frames[i] for i in indices])})
        seconds.append(time.perf_counter() - start)
        if feature.shape != (768,) or not np.isfinite(feature).all():
            raise ValueError("invalid feature")
        features.append(feature)
    path = args.output / "features.npy"
    np.save(path, np.stack(features), allow_pickle=False)
    report.update(
        status="PASS_RUNTIME_ONLY",
        windows=len(features),
        window_seconds=seconds,
        dense_seconds=sum(seconds),
        feature_bytes=path.stat().st_size,
        feature_sha256=sha256_file(path),
        indices=row["windows"],
        checkpoint_sha256=CHECKPOINT_SHA256,
        encoder_source_revision=SOURCE_COMMIT,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
