"""Serve a bounded local request batch from one pinned visual parameter tree."""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    _decode_video,
    _preprocess,
    sha256_file,
)
from tiny_omni_decision.shared_visual_encoder import SharedVisualEncoder
from tiny_omni_decision.token_transport import write_tokens
from tiny_omni_decision.visual_requests import VisualRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hold-for-release", action="store_true")
    args = parser.parse_args()
    if args.report.exists() or args.requests.stat().st_size > 1024 * 1024:
        raise ValueError("fresh report and bounded requests required")
    requests = [VisualRequest(**r) for r in json.loads(args.requests.read_text(encoding="utf-8"))]
    if not 1 <= len(requests) <= 32 or len({r.key for r in requests}) != len(requests):
        raise ValueError("1..32 unique request keys required")
    source = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
    weights = source.parent.parent / "videoprism-base-f16r288.npz"
    if sha256_file(weights) != CHECKPOINT_SHA256:
        raise ValueError("weights changed")
    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("source revision changed")
    stream = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(source))
        import jax
        import numpy as np
        import psutil
        import tensorflow as tf
        from videoprism import models as vp

        from scripts.extract_frozen_videoprism_clevr4_features import _preprocess_rgb_image

        if not all(d.platform == "cpu" for d in jax.devices()):
            raise ValueError("CPU profile required")
        model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(weights)
        )
        encoder = SharedVisualEncoder(model, variables, compile_fn=jax.jit)
        records = []
        for request in requests:
            started = time.perf_counter()
            for asset in request.assets:
                if sha256_file(Path(asset.path)) != asset.sha256:
                    raise ValueError("raw media hash changed")
            if request.modality == "image":
                frames = np.stack(
                    [
                        _preprocess_rgb_image(
                            tf.io.decode_png(tf.io.read_file(a.path), channels=3), tf
                        )[:1]
                        for a in request.assets
                    ]
                )
            else:
                frames = _preprocess(_decode_video(Path(request.assets[0].path)), tf)[None]
            features = encoder.encode(
                frames,
                modality=request.modality,
                representation="tokens" if request.modality == "video" else None,
            )
            if features.shape != request.output_shape():
                raise ValueError("profile output differs")
            write_tokens(stream, request.key, features)
            records.append(
                dict(
                    key=request.key,
                    shape=list(features.shape),
                    preprocess_encode_send_seconds=time.perf_counter() - started,
                )
            )
        assert sha256_file(weights) == CHECKPOINT_SHA256
        if args.hold_for_release and sys.stdin.readline().strip() != "release":
            raise RuntimeError("missing parent release")
        args.report.write_text(
            json.dumps(
                dict(
                    status="PASS_SHARED_WORKER",
                    pid=os.getpid(),
                    parameters=sum(v.size for v in jax.tree_util.tree_leaves(variables)),
                    weights_sha256=CHECKPOINT_SHA256,
                    source=SOURCE_COMMIT,
                    request_sha256=sha256_file(args.requests),
                    records=records,
                    memory=psutil.Process().memory_info()._asdict(),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
