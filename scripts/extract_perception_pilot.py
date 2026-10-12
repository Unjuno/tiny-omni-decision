"""Pinned frozen encoders on PTS-bounded train pilot windows."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.extract_starss23_features import (
    AUDIO_SHA,
    AUDIO_SOURCE,
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    load_encoder,
    sha256_file,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["audio", "video"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    args.source_root = r1 / (
        "efficientat-audit-v1/source" if args.modality == "audio" else "videoprism-base-r1/source"
    )
    args.checkpoint = r1 / (
        "efficientat-audit-v1/dymn04_as.pt"
        if args.modality == "audio"
        else "videoprism-base-f16r288.npz"
    )
    expected = AUDIO_SHA if args.modality == "audio" else CHECKPOINT_SHA256
    revision = AUDIO_SOURCE if args.modality == "audio" else SOURCE_COMMIT
    if (
        sha256_file(args.checkpoint) != expected
        or subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != revision
    ):
        raise ValueError("encoder revision mismatch")
    windows = Path("C:/CodexArtifacts/pretrained-reuse-r2/perception-windows-01/result.json")
    rows = json.loads(windows.read_text(encoding="utf-8"))["windows"]
    args.output.mkdir(exist_ok=False)
    report = dict(
        modality=args.modality,
        checkpoint_sha256=expected,
        source_revision=revision,
        windows_sha256=sha256_file(windows),
        training=False,
        features=[],
    )
    start = time.perf_counter()
    try:
        encode, shape = load_encoder(args, report)
        if args.modality == "video":
            report["preprocessing"] = (
                "PTS-selected PyAV RGB 288x288 windows; /255; frozen VideoPrism"
            )
        for row in rows:
            if row["status"] != "prepared":
                continue
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError("window changed")
            with np.load(path, allow_pickle=False) as data:
                feature = encode(data)
                repeat = encode(data)
            if feature.shape != shape or not np.isfinite(feature).all():
                raise ValueError("invalid feature")
            np.testing.assert_allclose(feature, repeat, atol=1e-6, rtol=1e-5)
            output = args.output / f"feature-{len(report['features']):03}.npy"
            np.save(output, feature, allow_pickle=False)
            report["features"].append(
                dict(
                    id=row["id"],
                    sha256=sha256_file(output),
                    shape=list(shape),
                    repeat_max_delta=float(abs(feature - repeat).max()),
                )
            )
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        report["preprocessing_scope"] = (
            "Perception PTS windows; inherited encoder normalization only"
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
