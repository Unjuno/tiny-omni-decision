"""Fixed-budget semantic-head versus pooled-feature audio comparison."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.extract_starss23_features import AUDIO_SHA
from scripts.fetch_starss23_frozen_media import write_json
from scripts.train_starss23_readouts import load_feature_matrix, standardize, train_condition


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    output = root / "starss-pretrained-head-comparison-01"
    manifest = root / "starss23-all-windows-v1"
    checkpoint = Path("C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt")
    if sha256_file(checkpoint) != AUDIO_SHA:
        raise ValueError("checkpoint mismatch")
    rows = read_manifest(manifest)
    manifest_sha = sha256_file(manifest / "result.json")
    features = load_feature_matrix(
        root / "starss23-audio-corpus-features-v1", "audio", rows, manifest_sha
    )
    config_path = Path("configs/pretrained_reuse/path_a_starss23_events.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    masks = {s: np.array([r["split"] == s for r in rows])
             for s in ("train", "calibration", "validation")}
    targets = np.array([[q["target"] for q in r["queries"]] for r in rows])
    output.mkdir(exist_ok=False)
    protocol = {
        "change": "pooled384 versus pretrained527 logits, each padded to1152",
        "mapping": "no manual label mapping; all12 original queries retained",
        "checkpoint_sha256": AUDIO_SHA, "manifest_sha256": manifest_sha,
        "config_sha256": sha256_file(config_path), "readout": config["readout"],
        "script_sha256": sha256_file(Path(__file__)), "seed": 17,
        "device": "cpu", "blind_audit": False, "torch": torch.__version__,
    }
    write_json(output / "protocol.json", protocol)
    validation_ids = [r["id"] for r in rows if r["split"] == "validation"]
    write_json(output / "validation_ids.json", validation_ids)
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    started = time.perf_counter()
    # Official mlp classifier: pool/flatten, linear, hardswish, eval dropout, linear.
    # Cached features are exactly its pooled/flattened input (not normalized).
    with torch.inference_mode():
        hidden = F.hardswish(F.linear(torch.from_numpy(features),
                                     state["classifier.2.weight"], state["classifier.2.bias"]))
        logits = F.linear(hidden, state["classifier.5.weight"], state["classifier.5.bias"]).numpy()
    if logits.shape != (len(rows), 527) or not np.isfinite(logits).all():
        raise ValueError("invalid pretrained output")
    np.save(output / "pretrained_logits.npy", logits)
    results = {}
    for name, values in (("pooled", features), ("pretrained_logits", logits)):
        scaled, mean, std = standardize(values, masks["train"])
        padded = np.zeros((len(rows), 1152), dtype=np.float32)
        padded[:, :scaled.shape[1]] = scaled
        dest = output / name
        dest.mkdir()
        np.savez(dest / "normalization.npz", mean=mean, std=std)
        _, _, _, results[name] = train_condition(padded, targets, masks, config["readout"], dest)
    results["wall_seconds"] = time.perf_counter() - started
    results["protocol"] = protocol
    write_json(output / "result.json", results)
    print(json.dumps({k: v["validation_calibrated"]["macro"] for k, v in results.items()
                      if k in ("pooled", "pretrained_logits")}))


if __name__ == "__main__":
    main()
