"""Cache fixed development episode features; sealed audit is not an option."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from tiny_omni_decision.quaternion_features import temporal_orientation_features
from tiny_omni_decision.quaternion_sto import parse_quaternion_sto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    split_path = root / "vidimu-frozen-split-01/split.json"
    if sha(split_path) != "230e4f9f9fb6218b90fea0894396bc96bf67a0951864301d0fe289f4a0457b2e":
        raise ValueError("split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    path = root / "vidimu-source-audit-02/dataset.zip"
    if sha(path) != split["archive_sha256"]:
        raise ValueError("archive changed")
    name = "training" if args.split == "train" else "validation"
    output = root / f"vidimu-{name}-sensor-features-01"
    output.mkdir(exist_ok=False)
    features, records = [], []
    with zipfile.ZipFile(path) as archive:
        for record in split["records"][args.split]:
            member = record["member"].removesuffix(".raw") + ".sto"
            parsed = parse_quaternion_sto(archive.read(member).decode())
            features.append(temporal_orientation_features(
                parsed["timestamps_ms"], parsed["values"]
            ))
            records.append({**record, "sensors": parsed["sensors"], "member": member})
    np.savez_compressed(output / "features.npz", values=np.stack(features).astype(np.float32))
    overlap = None
    if args.split == "validation":
        training_path = root / "vidimu-training-sensor-features-01/features.npz"
        if sha(training_path) != "84bead6d313839b7fa0949bebc4d66af2574c53f393ce717459359eb61496925":
            raise ValueError("training cache changed")
        with np.load(training_path, allow_pickle=False) as training:
            hashes = {hashlib.sha256(x.tobytes()).hexdigest() for x in training["values"]}
        overlap = sum(hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest() in hashes
                      for x in features)
        if overlap:
            raise ValueError("exact cross-split feature duplication")
    write(output / "records.json", records)
    write(output / "result.json", {
        "features_sha256": sha(output / "features.npz"),
        "records_sha256": sha(output / "records.json"),
        "shape": list(np.stack(features).shape), "split_sha256": sha(split_path),
        "all_initial_samples_retained": True, "split": args.split,
        "validation_read": args.split == "validation", "sealed_audit_read": False,
        "exact_training_feature_overlap": overlap,
        "representation": "8 equal-duration bins, mean rotation matrix entries, wxyz input",
    })
    print((output / "result.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
