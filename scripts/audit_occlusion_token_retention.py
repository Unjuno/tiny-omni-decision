"""Two train videos, boundary windows: token retention, not semantic quality."""

import json
import subprocess
import sys
from pathlib import Path

import av
import numpy as np

from scripts.extract_occlusion_dense_features import array_hash
from scripts.extract_starss23_features import CHECKPOINT_SHA256, SOURCE_COMMIT, sha256_file
from tiny_omni_decision.video_token_grid import token_grid


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
    checkpoint = source.parent.parent / "videoprism-base-f16r288.npz"
    output = root / "perception-occlusion-token-retention-01"
    report_path = root / "perception-occlusion-dense-features-01/result.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "PASS_FEATURE_EXTRACTION_NOT_QUALITY":
        raise ValueError("incomplete parent")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint changed")
    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("source changed")
    selected = [r for r in report["features"] if r["split"] == "train"][:2]
    manifest = json.loads((root / "perception-occlusion-train-media-01/manifest.json").read_text())
    media = {r["video_id"]: r for r in manifest["clips"]}
    output.mkdir(exist_ok=False)
    config = dict(
        parent_sha256=sha256_file(report_path),
        selected=[r["video_id"] for r in selected],
        training=False,
        quality_evaluation=False,
        script_sha256=sha256_file(Path(__file__)),
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    weights = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    forward = jax.jit(lambda x: model.apply(weights, x, train=False))
    results = []
    for row in selected:
        path = Path(media[row["video_id"]]["path"])
        if (
            sha256_file(path) != row["media_sha256"]
            or sha256_file(Path(row["path"])) != row["sha256"]
        ):
            raise ValueError("parent input changed")
        boundary = [0, len(row["plan"]["windows"]) - 1]
        wanted = {i for w in boundary for i in row["plan"]["windows"][w]}
        frames = {}
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    frames[i] = frame.reformat(width=288, height=288, format="rgb24").to_ndarray()
                if i >= max(wanted):
                    break
        pooled = np.load(row["path"], allow_pickle=False)
        for window in boundary:
            pixels = np.stack([frames[i] for i in row["plan"]["windows"][window]])
            if array_hash(pixels) != row["window_input_hashes"][window]:
                raise ValueError("prepared pixels changed")
            value = forward(jnp.asarray(pixels[None].astype(np.float32) / 255))
            tokens = np.asarray(value[0] if isinstance(value, tuple) else value)
            grid = token_grid(tokens)
            mean = tokens.mean(1)[0]
            np.testing.assert_allclose(mean, pooled[window], rtol=1e-5, atol=1e-6)
            target = output / f"{row['video_id']}-{window}.npy"
            np.save(target, grid, allow_pickle=False)
            # Reindexing tokens measures the information discarded by pooling;
            # it is not a physical image reflection or an object-localization test.
            mirrored = grid[:, :, ::-1]
            results.append(
                dict(
                    video_id=row["video_id"],
                    window=window,
                    shape=list(grid.shape),
                    sha256=sha256_file(target),
                    pooled_max_abs=float(np.max(np.abs(mean - pooled[window]))),
                    reindexed_token_relative_l2=float(
                        np.linalg.norm(grid - mirrored) / np.linalg.norm(grid)
                    ),
                    reindexed_mean_max_abs=float(
                        np.max(np.abs(grid.mean((0, 1, 2)) - mirrored.mean((0, 1, 2))))
                    ),
                )
            )
            print(row["video_id"], window, "PASS", flush=True)
    (output / "result.json").write_text(
        json.dumps(dict(status="PASS_TOKEN_RETENTION_NOT_SEMANTICS", records=results), indent=2)
    )


if __name__ == "__main__":
    main()
