"""Admission for the fixed VideoSAUR development validation endpoint."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_correspondence_validation import sha
from scripts.slot_feature_sampling import sample_indices

ROOT = Path("C:/CodexArtifacts/prevalided-reuse-r2")
PAIR_SHA = "30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2"
MEDIA_SHA = "213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546"


PROTOCOL_SHA = "be99f0ff5e2f3d4d3037ea0f33e46da499ced1db904a78718e60cc9ade911ca4"


def validate_feature(row, value, media):
    if row["split"] != "valid" or row["media_sha256"] != media["sha256"]:
        raise ValueError("feature media binding mismatch")
    if value.shape != (224, 129) or not np.isfinite(value).all():
        raise ValueError("invalid feature shape/value")
    expected = sample_indices(row["frame_count"])
    if row["frame_indices"] != expected:
        raise ValueError("frame sampling changed")
    times = np.repeat(np.asarray(expected, dtype=np.float32) / (row["frame_count"] - 1), 7)
    if not np.array_equal(value[:, -1], times):
        raise ValueError("time features changed")


def load_valid(folder):
    ledger = json.loads((folder / "result.json").read_text())
    if ledger["status"] != "PASS_FIXED_VALID_SLOT_EXTRACTION" or not ledger["weights_unchanged"]:
        raise ValueError("complete frozen extraction required")
    protocol_path = folder / "protocol.json"
    if sha(protocol_path) != ledger["protocol_sha256"]:
        raise ValueError("protocol changed")
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("unexpected experiment protocol")
    protocol = json.loads(protocol_path.read_text())
    if protocol["manifest_sha256"] != MEDIA_SHA or protocol["pair_sha256"] != PAIR_SHA:
        raise ValueError("source corpus changed")
    pair_path = ROOT / "perception-occlusion-pairs-01/valid.json"
    manifest_path = ROOT / "perception-occlusion-valid-media-01/manifest.json"
    if sha(pair_path) != PAIR_SHA or sha(manifest_path) != MEDIA_SHA:
        raise ValueError("local corpus changed")
    pairs = json.loads(pair_path.read_text())["pairs"]
    media = {r["video_id"]: r for r in json.loads(manifest_path.read_text())["clips"]}
    if len(pairs) != 257 or len(media) != 257 or len({r["sha256"] for r in media.values()}) != 257:
        raise ValueError("incomplete/duplicate corpus")
    if any([q["condition"] for q in p["questions"]] != ["initial", "final"] for p in pairs):
        raise ValueError("condition order changed")
    plan = {r["video_id"]: r for r in protocol["plan"]}
    if set(plan) != set(media):
        raise ValueError("feature plan incomplete")
    bank = {}
    hashes = {
        str(folder / "result.json"): sha(folder / "result.json"),
        str(protocol_path): sha(protocol_path),
    }
    for row in ledger["features"]:
        path = Path(row["path"])
        key = row["video_id"]
        if key in bank or key not in media or sha(path) != row["sha256"]:
            raise ValueError("duplicate/missing/changed feature")
        value = np.load(path, allow_pickle=False)
        if row["frame_count"] != plan[key]["frame_count"]:
            raise ValueError("frame count differs from frozen plan")
        validate_feature(row, value, media[key])
        bank[key] = torch.from_numpy(value)
        hashes[str(path)] = sha(path)
    if set(bank) != {p["video_id"] for p in pairs}:
        raise ValueError("all 257 clips required")
    text_path = ROOT / "perception-occlusion-text-features-01/result.json"
    text = json.loads(text_path.read_text())
    if text["status"] != "PASS_FROZEN_TEXT_FEATURES" or text["pair_hashes"]["valid"] != PAIR_SHA:
        raise ValueError("text provenance changed")
    row = text["features"]["valid"]
    path = Path(row["path"])
    if sha(path) != row["sha256"]:
        raise ValueError("text features changed")
    expected = [
        dict(video_id=p["video_id"], condition=q["condition"], target=q["target"])
        for p in pairs
        for q in p["questions"]
    ]
    x = np.load(path, allow_pickle=False)
    if row["rows"] != expected or x.shape != (514, 3, 1536) or not np.isfinite(x).all():
        raise ValueError("text routing/shape/value mismatch")
    hashes[str(text_path)] = sha(text_path)
    hashes[str(path)] = sha(path)
    return (
        bank,
        dict(
            pairs=pairs,
            text=torch.from_numpy(x),
            targets=torch.tensor([r["target"] for r in expected]),
        ),
        hashes,
    )

