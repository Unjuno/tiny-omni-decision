"""Pinned local CPU plumbing audit; output directory must be new."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

root = Path("C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-audit-01")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
output = parser.parse_args().output
output.mkdir(parents=True, exist_ok=False)
assert hashlib.sha256((root / "modeling_finetune.py").read_bytes()).hexdigest() == (
    "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
)
assert hashlib.sha256((root / "vit_s_k710_dl_from_giant.pth").read_bytes()).hexdigest() == (
    "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
)
sys.path.insert(0, "C:/CodexArtifacts/pretrained-reuse-r1/python-deps")
spec = importlib.util.spec_from_file_location(
    "videomae_audit_source", root / "modeling_finetune.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
torch.set_num_threads(4)
state = torch.load(root / "vit_s_k710_dl_from_giant.pth", map_location="cpu", weights_only=True)[
    "module"
]
model = mod.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
print(model.load_state_dict(state, strict=True), flush=True)
path = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-train-media-01/video_10186.mp4"
)
assert (
    hashlib.sha256(path.read_bytes()).hexdigest()
    == "6e61d8f411114ef7f052a5bea40bd51b4161274bb0d799cd158658a14b6d4df7"
)
ids = np.linspace(0, 650, 16).astype(int).tolist()
frames = {}
with av.open(str(path)) as c:
    for i, f in enumerate(c.decode(video=0)):
        if i in ids:
            frames[i] = f.reformat(width=224, height=224, format="rgb24").to_ndarray()
        if i >= 650:
            break
pixels = np.stack([frames[i] for i in ids])
x = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).float()[None] / 255
x = (x - 0.5) / 0.5
start = time.perf_counter()
with torch.inference_mode():
    y = model(x)
    f = model.forward_features(x)
    again = model(x)
elapsed = time.perf_counter() - start
assert torch.isfinite(y).all() and torch.isfinite(f).all()
torch.testing.assert_close(y, again, rtol=0, atol=0)
result = {
    "status": "PASS_STRICT_LOAD_REAL_VIDEO_FORWARD_NOT_QUALITY",
    "parameters": sum(p.numel() for p in model.parameters()),
    "logits_shape": list(y.shape),
    "features_shape": list(f.shape),
    "seconds_three_forwards": elapsed,
    "repeat_exact": True,
    "torch": torch.__version__,
    "device": "cpu",
    "dtype": "float32",
    "frame_ids": ids,
    "preprocessing": (
        "diagnostic full resize224, RGB /255 then mean/std0.5; "
        "not official multi-crop benchmark"
    ),
    "source_file_sha256": hashlib.sha256((root / "modeling_finetune.py").read_bytes()).hexdigest(),
    "checkpoint_sha256": hashlib.sha256(
        (root / "vit_s_k710_dl_from_giant.pth").read_bytes()
    ).hexdigest(),
}
with (output / "runtime.json").open("x") as h:
    json.dump(result, h, indent=2)
np.save(output / "features.npy", f.numpy(), allow_pickle=False)
print(json.dumps(result), flush=True)
