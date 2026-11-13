"""Oracle-initialized, two-train-video token correspondence; no fitting."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import av
import numpy as np

from scripts.extract_occlusion_dense_features import PINS, array_hash
from scripts.extract_starss23_features import CHECKPOINT_SHA256, SOURCE_COMMIT, sha256_file
from tiny_omni_decision.video_token_grid import token_grid


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
    checkpoint = source.parent.parent / "videoprism-base-f16r288.npz"
    archive = root / "perception-test-source-audit-01/train_annotations.zip"
    manifest = root / "perception-occlusion-train-media-01/manifest.json"
    if sha256_file(archive) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("annotations changed")
    if sha256_file(manifest) != PINS["train"] or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("manifest/checkpoint changed")
    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("source changed")
    with zipfile.ZipFile(archive) as z:
        annotations = json.loads(z.read("all_train.json"))
    plans = []
    for row in json.loads(manifest.read_text())["clips"][:2]:
        tracks = [t for t in annotations[row["video_id"]]["object_tracking"] if t["is_occluder"]]
        if len(tracks) != 1:
            raise ValueError("ambiguous occluder track")
        track = tracks[0]
        points = [
            (i, b)
            for i, b, m in zip(
                track["frame_ids"], track["bounding_boxes"], track["is_masked"], strict=True
            )
            if not m and i < row["cut_frame"]
        ]
        if len(points) < 8:
            raise ValueError("insufficient exact observations")
        selected = [points[i] for i in np.linspace(0, len(points) - 1, 8).astype(int)]
        plans.append(dict(**row, track_id=track["id"], points=selected))
    output = root / "perception-occlusion-track-tokens-01"
    output.mkdir(exist_ok=False)
    config = dict(
        plans=plans,
        script_sha256=sha256_file(Path(__file__)),
        checkpoint_sha256=CHECKPOINT_SHA256,
        source_revision=SOURCE_COMMIT,
        training=False,
        scope="oracle-initialized offline correspondence; not Decision quality",
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
    for plan in plans:
        path = Path(plan["path"])
        if sha256_file(path) != plan["sha256"]:
            raise ValueError("media changed")
        ids = [p[0] for p in plan["points"]]
        frames = {}
        with av.open(str(path)) as c:
            for i, frame in enumerate(c.decode(video=0)):
                if i in ids:
                    frames[i] = frame.reformat(width=288, height=288, format="rgb24").to_ndarray()
                if i >= max(ids):
                    break
        pixels = np.stack([frames[i] for i in ids])
        value = forward(jnp.asarray(pixels[None].astype(np.float32) / 255))
        grid = token_grid(np.asarray(value[0] if isinstance(value, tuple) else value))
        np.save(output / (plan["video_id"] + ".npy"), grid, allow_pickle=False)
        boxes = np.array([p[1] for p in plan["points"]])
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        xy = np.minimum((centers[0] * 16).astype(int), 15)
        query = grid[0, xy[1], xy[0]].astype(float)
        query /= np.linalg.norm(query)
        predictions = []
        for index in range(1, 8):
            cells = grid[index].reshape(256, 768).astype(float)
            cells /= np.maximum(np.linalg.norm(cells, axis=1, keepdims=True), 1e-12)
            best = int(np.argmax(cells @ query))
            point = np.array([(best % 16 + 0.5) / 16, (best // 16 + 0.5) / 16])
            predictions.append(
                dict(
                    frame_id=ids[index],
                    predicted_xy=point.tolist(),
                    center_error=float(np.linalg.norm(point - centers[index])),
                    static_error=float(np.linalg.norm(centers[0] - centers[index])),
                    point_in_box=bool(
                        np.all(point >= boxes[index, :2]) and np.all(point <= boxes[index, 2:])
                    ),
                )
            )
        results.append(
            dict(
                video_id=plan["video_id"], input_sha256=array_hash(pixels), predictions=predictions
            )
        )
        print(plan["video_id"], "complete", flush=True)
    (output / "result.json").write_text(
        json.dumps(dict(status="COMPLETED_DIAGNOSTIC", results=results), indent=2)
    )


if __name__ == "__main__":
    main()
