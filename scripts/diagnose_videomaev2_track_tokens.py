"""Two fixed train tracks, oracle initial point, frozen offline token matching."""

import importlib.util
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch

from scripts.audit_correspondence_validation import sha, write
from scripts.extract_occlusion_videomaev2_features import MODEL, SOURCE_SHA, WEIGHT_SHA, array_hash


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    config = root / "perception-occlusion-track-tokens-01/config.json"
    if sha(config) != "f79f6d1426c63e57eec73ab1c4a793dc11b3d745213a035e6661ef81f0b76523":
        raise ValueError("track plan changed")
    plans = json.loads(config.read_text())["plans"]
    source = MODEL / "modeling_finetune.py"
    weights = MODEL / "vit_s_k710_dl_from_giant.pth"
    if sha(source) != SOURCE_SHA or sha(weights) != WEIGHT_SHA:
        raise ValueError("model changed")
    output = root / "perception-occlusion-videomaev2-track-tokens-01"
    output.mkdir(exist_ok=False)
    sys.path.insert(0, "C:/CodexArtifacts/pretrained-reuse-r1/python-deps")
    spec = importlib.util.spec_from_file_location("track_videomae", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    model = module.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["module"])
    capture = {}

    def hook(_module, _inputs, value):
        capture["tokens"] = value.detach()

    handle = model.blocks[-1].register_forward_hook(hook)
    result = dict(
        plan_sha256=sha(config),
        source_sha256=SOURCE_SHA,
        checkpoint_sha256=WEIGHT_SHA,
        script_sha256=sha(Path(__file__)),
        scope="oracle-initialized offline train tracking, not Decision quality",
        results=[],
    )
    with torch.inference_mode():
        for plan in plans:
            path = Path(plan["path"])
            if sha(path) != plan["sha256"]:
                raise ValueError("media changed")
            ids = [p[0] for p in plan["points"]]
            frames = {}
            with av.open(str(path)) as c:
                for i, frame in enumerate(c.decode(video=0)):
                    if i in ids:
                        frames[i] = frame.reformat(
                            width=224, height=224, format="rgb24"
                        ).to_ndarray()
                    if i >= max(ids):
                        break
            pixels = np.repeat(np.stack([frames[i] for i in ids]), 2, axis=0)
            x = torch.from_numpy(pixels).permute(3, 0, 1, 2).float()[None] / 255
            pooled = model.forward_features((x - 0.5) / 0.5)
            tokens = capture["tokens"]
            torch.testing.assert_close(pooled, model.fc_norm(tokens.mean(1)), rtol=0, atol=0)
            grid = tokens.numpy().reshape(8, 14, 14, 384)
            np.save(output / f"{plan['video_id']}.npy", grid, allow_pickle=False)
            boxes = np.asarray([p[1] for p in plan["points"]])
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            xy = np.minimum((centers[0] * 14).astype(int), 13)
            query = grid[0, xy[1], xy[0]].astype(float)
            query /= np.linalg.norm(query)
            static = (xy + 0.5) / 14
            predictions = []
            for index in range(1, 8):
                cells = grid[index].reshape(196, 384).astype(float)
                cells /= np.maximum(np.linalg.norm(cells, axis=1, keepdims=True), 1e-12)
                best = int(np.argmax(cells @ query))
                point = np.array([(best % 14 + 0.5) / 14, (best // 14 + 0.5) / 14])
                predictions.append(
                    dict(
                        frame_id=ids[index],
                        predicted_xy=point.tolist(),
                        center_error=float(np.linalg.norm(point - centers[index])),
                        static_grid_error=float(np.linalg.norm(static - centers[index])),
                        point_in_box=bool(
                            np.all(point >= boxes[index, :2]) and np.all(point <= boxes[index, 2:])
                        ),
                    )
                )
            result["results"].append(
                dict(
                    video_id=plan["video_id"],
                    pixel_sha256=array_hash(pixels),
                    token_sha256=sha(output / f"{plan['video_id']}.npy"),
                    predictions=predictions,
                    pooled_exact=True,
                )
            )
    handle.remove()
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
