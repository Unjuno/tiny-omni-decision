"""Extract the fixed VIDIMU development episodes; never read audit payloads."""

import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    _preprocess,
    sha256_file,
)
from tiny_omni_decision.video_frame_timestamps import decode_endpoint_frames


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    mapping = root / "vidimu-video-mapping-02/mapping.json"
    archive_path = root / "vidimu-small-video-audit-01/videosmallsize.zip"
    expected = {
        mapping: "5b87e50675cc5234341dbc3ec7bc9febe6abfaa0130570e7a1bf1a4c4d8b5532",
        archive_path: "5224354912d0e72b0e6307c344ee6386d27bbd48fd26352d820a091c99fea9b7",
    }
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    checkpoint = r1 / "videoprism-base-f16r288.npz"
    expected[checkpoint] = CHECKPOINT_SHA256
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"hash mismatch: {path}")
    source = r1 / "videoprism-base-r1/source"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError("upstream source changed")
    records = json.loads(mapping.read_text())
    if set(records) != {"train", "validation"}:
        raise ValueError("development splits only")
    if [len(records[x]) for x in ("train", "validation")] != [130, 39]:
        raise ValueError("fixed trial counts changed")
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    out = root / "vidimu-video-features-01"
    out.mkdir(exist_ok=False)
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda x: model.apply(variables, x, train=False))
    started = time.perf_counter()
    media_hashes = {}
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as tmp:
        for split in ("train", "validation"):
            values, details = [], []
            for index, row in enumerate(records[split]):
                start = time.perf_counter()
                media = Path(tmp) / "input.mp4"
                media.write_bytes(archive.read(row["video_member"]))
                digest = sha256_file(media)
                if digest in media_hashes:
                    raise ValueError("duplicate development video bytes")
                media_hashes[digest] = split
                frames, timestamps = decode_endpoint_frames(media)
                result = encode(jnp.asarray(_preprocess(frames, tf)[None, ...]))
                tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
                if tokens.shape != (1, 2048, 768):
                    raise ValueError(f"unexpected token shape: {tokens.shape}")
                features = tokens.reshape(8, 16, 16, 768).mean(axis=(1, 2))
                if not np.isfinite(features).all():
                    raise ValueError("nonfinite features")
                values.append(features.astype(np.float32))
                details.append({**row, "media_sha256": digest, **timestamps,
                                "seconds": time.perf_counter() - start})
                with (out / "progress.jsonl").open("a") as log:
                    log.write(json.dumps({"split": split, "completed": index + 1,
                                          **details[-1]}) + "\n")
                print(f"{split} {index + 1}/{len(records[split])}", flush=True)
            np.savez_compressed(out / f"{split}.npz", values=np.stack(values))
            (out / f"{split}-records.json").write_text(json.dumps(details, indent=2) + "\n")
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  text=True).strip(),
        "script_sha256": sha256_file(Path(__file__)),
        "input_hashes": {str(p): h for p, h in expected.items()},
        "feature_hashes": {s: sha256_file(out / f"{s}.npz") for s in records},
        "parameters": sum(x.size for x in jax.tree_util.tree_leaves(variables)),
        "devices": [str(x) for x in jax.devices()],
        "seconds": time.perf_counter() - started,
        "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", None),
        "sealed_audit_read": False, "sensor_clock_alignment_asserted": False,
    }
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
