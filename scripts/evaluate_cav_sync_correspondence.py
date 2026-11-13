"""Frozen real training-input smoke; no decision quality claim."""

import hashlib
import importlib.util
import json
import sys
import time
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

root = Path("C:/CodexArtifacts/pretrained-reuse-r1")
source = root / "cav-mae-sync-source-audit-01"
out = root / "cav-mae-sync-correspondence-valid-01"
out.mkdir(exist_ok=False)
sys.path.insert(0, str(root / "cav-mae-sync-deps-01"))
package = types.ModuleType("cav_structure")
package.__path__ = [str(source)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    "cav_structure.model", source / "src__models__cav_mae_sync.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
torch.manual_seed(17)
model = (
    module.CAVMAE(
        audio_length=416,
        modality_specific_depth=11,
        num_register_tokens=8,
        cls_token=True,
        total_frame=4,
        contrastive_heads=False,
    )
    .eval()
    .requires_grad_(False)
)
checkpoint = root / "cav-mae-sync-weight-audit-01/checkpoint.pth"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
    "a035b04cf615fe4c85d6aa1e74b26d8223a85945957664e66abea16e183629bd"
)
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
assert all(k.startswith("module.") for k in state)
model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
del state
sys.path.append(str(root / "python-deps"))


import torchaudio  # noqa: E402 - isolated optional dependency path above

windows = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-correspondence-development-windows-01"
)
manifest = json.loads((windows / "result.json").read_text())
preprocess = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
features_a, features_v, records = [], [], []
start = time.perf_counter()
model.cuda()
torch.cuda.reset_peak_memory_stats()
for row in manifest["windows"]:
    if row["split"] != "valid":
        continue
    path = Path(row["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    data = np.load(path, allow_pickle=False)
    rate = row["audio_sample_rate"]
    frame_index = len(data["video"]) // 2
    stamp = float(Fraction(row["selected_pts"][frame_index]))
    center = stamp - row["interval_us"][0] / 1e6
    count = round(4.175 * rate)  # 416 x 10ms hops plus 25ms final frame
    assert len(data["audio"]) >= count
    begin = min(max(0, round(center * rate) - count // 2), len(data["audio"]) - count)
    waveform = torch.from_numpy(data["audio"][begin : begin + count].T.copy())
    waveform = waveform - waveform.mean()
    bank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    assert bank.shape == (416, 128), bank.shape
    bank = (bank + 5.081) / 4.4849
    image = preprocess(Image.fromarray(data["video"][frame_index]))
    with torch.inference_mode():
        value = model.forward_feat(bank[None].cuda(), image[None].cuda())
        repeat = model.forward_feat(bank[None].cuda(), image[None].cuda())
    assert all(torch.isfinite(x).all() for x in value)
    assert all(torch.equal(x, y) for x, y in zip(value, repeat, strict=True))
    features_a.append(value[2].reshape(768).cpu().numpy())
    features_v.append(value[3].reshape(768).cpu().numpy())
    records.append(
        dict(
            id=row["id"],
            input_sha256=row["sha256"],
            frame_index=frame_index,
            frame_pts=stamp,
            audio_bounds=[begin, begin + count],
            sample_rate=rate,
        )
    )
np.savez(out / "features.npz", audio=np.stack(features_a), video=np.stack(features_v))
report = dict(
    status="FIXED_VALIDATION_ZERO_SHOT",
    records=records,
    finite=True,
    exact_repeat=True,
    elapsed_seconds=time.perf_counter() - start,
    feature_sha256=hashlib.sha256((out / "features.npz").read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256((windows / "result.json").read_bytes()).hexdigest(),
    preprocessing="Centered 4.175s at retained frame; Kaldi416,128; pinned normalization",
    pretrained_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
)
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


from scripts.train_sensor_subjects import metrics  # noqa: E402

route_path = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-correspondence-valid-audit-01/valid-routing.json"
)
assert (
    hashlib.sha256(route_path.read_bytes()).hexdigest()
    == "03d389b0eda847b4d16e0205a7c285ad78ebecfb6405fde1c85c0ddfd569d395"
)
routes = json.loads(route_path.read_text())
a = np.stack(features_a)
v = np.stack(features_v)
a /= np.linalg.norm(a, axis=1, keepdims=True)
v /= np.linalg.norm(v, axis=1, keepdims=True)
index = {r["id"]: i for i, r in enumerate(records)}
assert len(index) == len(routes) == 200
logits = np.array(
    [
        [a[index[r["audio_video_id"]]] @ v[index[c]] for c in r["candidate_video_ids"]]
        for r in routes
    ]
)
prob = np.exp(logits - logits.max(1, keepdims=True))
prob /= prob.sum(1, keepdims=True)
y = np.array([r["target_index"] for r in routes])
report["metrics"] = metrics(prob, y)
report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
report["temperature"] = 1.0
report["routing_sha256"] = hashlib.sha256(route_path.read_bytes()).hexdigest()
np.savez(out / "predictions.npz", probabilities=prob, targets=y, logits=logits)
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report["metrics"], indent=2))
