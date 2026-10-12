"""Frozen VideoMAEv2 features on the pinned prior experiment's exact windows."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

from scripts.audit_correspondence_validation import sha, write
from scripts.train_occlusion_readout import PINS as PAIR_PINS

PINS = {
    "train": "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",
    "valid": "213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546",
}


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
MODEL = Path("C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-audit-01")
PLAN_SHA = "b7d477c34e8faf047804c77750e154d59afed3568a190eaf7cdef5380bc97fb8"
WEIGHT_SHA = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
SOURCE_SHA = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / "perception-occlusion-dense-features-01/result.json"
    if sha(plan_path) != PLAN_SHA:
        raise ValueError("prior sampling plan changed")
    old = json.loads(plan_path.read_text())
    if old["status"] != "PASS_FEATURE_EXTRACTION_NOT_QUALITY":
        raise ValueError("prior extraction incomplete")
    media = {}
    for split, pin in PINS.items():
        path = ROOT / f"perception-occlusion-{split}-media-01/manifest.json"
        pairs = ROOT / f"perception-occlusion-pairs-01/{split}.json"
        if sha(path) != pin or sha(pairs) != PAIR_PINS[split]:
            raise ValueError("input provenance changed")
        for row in json.loads(path.read_text())["clips"]:
            if row["video_id"] in media:
                raise ValueError("duplicate media identity")
            media[row["video_id"]] = dict(row, split=split)
    if len(media) != 356 or {r["video_id"] for r in old["features"]} != set(media):
        raise ValueError("missing or extra video")
    source, weights = MODEL / "modeling_finetune.py", MODEL / "vit_s_k710_dl_from_giant.pth"
    if sha(source) != SOURCE_SHA or sha(weights) != WEIGHT_SHA:
        raise ValueError("model changed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="RUNNING",
        pair_hashes=PAIR_PINS,
        manifest_hashes=PINS,
        prior_plan_sha256=PLAN_SHA,
        feature_dim=384,
        checkpoint_sha256=WEIGHT_SHA,
        source_file_sha256=SOURCE_SHA,
        script_sha256=sha(Path(__file__)),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        started_unix=time.time(),
        device="cpu",
        threads=4,
        dtype="float32",
        preprocessing="same 8 frame IDs repeated twice each; RGB resize224; /255, mean/std0.5",
        training=False,
        features=[],
    )
    try:
        sys.path.insert(0, "C:/CodexArtifacts/pretrained-reuse-r1/python-deps")
        spec = importlib.util.spec_from_file_location("videomaev2_extraction", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.set_num_threads(4)
        model = module.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["module"])
        report["parameters"] = sum(p.numel() for p in model.parameters())
        hashes = {"train": set(), "valid": set()}
        with torch.inference_mode():
            for prior in old["features"]:
                tick = time.perf_counter()
                row = media[prior["video_id"]]
                path = Path(row["path"])
                if row["split"] != prior["split"] or sha(path) != prior["media_sha256"]:
                    raise ValueError("media or split changed")
                plan = prior["plan"]
                wanted = set(plan["indices"])
                if max(wanted) >= row["cut_frame"]:
                    raise ValueError("reveal cut violated")
                frames = {}
                with av.open(str(path)) as container:
                    for i, frame in enumerate(container.decode(video=0)):
                        if i in wanted:
                            frames[i] = frame.reformat(
                                width=224, height=224, format="rgb24"
                            ).to_ndarray()
                        if i >= max(wanted):
                            break
                values, inputs = [], []
                for indices in plan["windows"]:
                    if len(indices) != 8:
                        raise ValueError("unexpected temporal width")
                    pixels = np.repeat(np.stack([frames[i] for i in indices]), 2, axis=0)
                    key = array_hash(pixels)
                    other = "valid" if row["split"] == "train" else "train"
                    if key in hashes[other]:
                        raise ValueError("cross split prepared content overlap")
                    hashes[row["split"]].add(key)
                    inputs.append(key)
                    x = torch.from_numpy(pixels).permute(3, 0, 1, 2).float()[None] / 255
                    feature = model.forward_features((x - 0.5) / 0.5)[0].numpy()
                    if feature.shape != (384,) or not np.isfinite(feature).all():
                        raise ValueError("invalid feature")
                    values.append(feature)
                target = args.output / f"{row['split']}-{row['video_id']}.npy"
                values = np.stack(values)
                np.save(target, values, allow_pickle=False)
                np.testing.assert_array_equal(values, np.load(target, allow_pickle=False))
                report["features"].append(
                    dict(
                        video_id=row["video_id"],
                        split=row["split"],
                        media_sha256=row["sha256"],
                        path=str(target),
                        sha256=sha(target),
                        shape=list(values.shape),
                        window_input_hashes=inputs,
                        plan=plan,
                        seconds=time.perf_counter() - tick,
                    )
                )
                write(args.output / "result.json", report)
                print(f"{len(report['features'])}/356 {row['video_id']}", flush=True)
        report.update(status="PASS_FEATURE_EXTRACTION_NOT_QUALITY", prepared_cross_split_overlap=0)
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["ended_unix"] = time.time()
        write(args.output / "result.json", report)


if __name__ == "__main__":
    main()
