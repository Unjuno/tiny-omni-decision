"""Train-only, hash-bound original QA bank; never open official validation arrays."""

import json
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha
from tiny_omni_decision.fine_qa_inputs import paired_subset, scene_split

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PINS = {
    "fine-train-context-all-01/result.json": "06fb6205e8baec945780878e0c1bb4999ad326760ffd8b37b3132b65a8d0c147",  # noqa: E501
    "fine-train-context-all-01/config.json": "602fa6e0b21d352b791b5aa1dbbce0a0b3262bb868dbd13ce96491b86b03dd02",  # noqa: E501
    "fine-train-context-independent-audit-01.json": "88bf740507f02abcc0102ad4c6e6ab0396e78fe0f976a2c8f72dee56aa2cd300",  # noqa: E501
    "spatial-context-corpus-01/result.json": "a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da",  # noqa: E501
    "spatial-context-corpus-01/config.json": "d17c1f1f9e55f17d58983603c1c8bf27ac171bed9d2ee04108535c84b15e35ff",  # noqa: E501
    "perception-occlusion-trajectory-features-01/result.json": "30910583657724619e82334fd7effca128b9e5415834422c8f4804c75ffe3d68",  # noqa: E501
    "perception-occlusion-pairs-01/train.json": "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7",  # noqa: E501
    "perception-occlusion-text-features-01/result.json": "3363b22508bae700c1bbb40ae7c328ba8beab12cd8f3e47a884061944ea48448",  # noqa: E501
    "perception-occlusion-train-media-01/manifest.json": "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",  # noqa: E501
    "spatial-readout-full-01/train-labels.npz": "6d0df64cdd81456591d66e95dea9f79064d4816549fc38722adcdec3f58f54d5",  # noqa: E501
}


def train_rows(rows, identities):
    selected = [r for r in rows if r["split"] == "train"]
    bank = {r["video_id"]: r for r in selected}
    if len(bank) != len(selected) or set(bank) != set(identities):
        raise ValueError("complete unique train identities required")
    return bank


def verify_binding(coarse, fine, point, media):
    for key in (
        "video_id",
        "split",
        "media_sha256",
        "original_frames",
        "frame_indices",
        "selected_rgb_sha256",
    ):
        if coarse[key] != fine[key]:
            raise ValueError("coarse/fine native source binding changed")
    if (
        coarse["split"] != "train"
        or point["split"] != "train"
        or coarse["video_id"] != point["video_id"]
        or coarse["video_id"] != media["video_id"]
        or coarse["media_sha256"] != media["sha256"]
        or point["media_sha256"] != media["sha256"]
        or coarse["original_frames"] != point["frames"]
        or media["cut_frame"] not in {-1, point["frames"]}
        or point["frame_indices"] != coarse["frame_indices"]
    ):
        raise ValueError("point/spatial/media native time binding changed")


def native_array(row, shape, dtype, hashes):
    path = Path(row["path"])
    if (
        sha(path) != row["sha256"]
        or row["shape"] != list(shape)
        or ("file_bytes" in row and row["file_bytes"] != path.stat().st_size)
    ):
        raise ValueError("native array hash/schema changed")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError("finite native array dtype/shape required")
    if "dtype" in row and row["dtype"] != str(value.dtype):
        raise ValueError("declared dtype mismatch")
    hashes[str(path)] = row["sha256"]
    return value


def load():
    hashes = {}
    reports = {}
    for name, pin in PINS.items():
        path = ROOT / name
        if sha(path) != pin:
            raise ValueError(f"pinned source changed: {name}")
        hashes[str(path)] = pin
        if path.suffix == ".json":
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    expected_status = {
        "fine-train-context-all-01/result.json": "PASS_ALL_TRAIN_FINE_SPATIAL_CORPUS_NOT_QUALITY",
        "spatial-context-corpus-01/result.json": "PASS_COMPLETE_SPATIAL_CONTEXT_CORPUS_NOT_QUALITY",
        "perception-occlusion-trajectory-features-01/result.json": "PASS_FEATURE_EXTRACTION_NOT_QUALITY",  # noqa: E501
        "perception-occlusion-text-features-01/result.json": "PASS_FROZEN_TEXT_FEATURES",
        "fine-train-context-independent-audit-01.json": "PASS_INDEPENDENT_ALL_TRAIN_FINE_INPUT_AUDIT_NOT_QUALITY",  # noqa: E501
    }
    for name, status in expected_status.items():
        if reports[name]["status"] != status:
            raise ValueError("input admission status mismatch")
    pairs = reports["perception-occlusion-pairs-01/train.json"]["pairs"]
    identities = [p["video_id"] for p in pairs]
    if len(pairs) != 99 or len(set(identities)) != 99:
        raise ValueError("all99 original train videos required")
    media_rows = reports["perception-occlusion-train-media-01/manifest.json"]["clips"]
    media = {r["video_id"]: r for r in media_rows}
    if len(media) != 99 or set(media) != set(identities) or len(media_rows) != 99:
        raise ValueError("all99 unique train media required")
    coarse = train_rows(reports["spatial-context-corpus-01/result.json"]["features"], identities)
    fine = train_rows(reports["fine-train-context-all-01/result.json"]["features"], identities)
    points = train_rows(
        reports["perception-occlusion-trajectory-features-01/result.json"]["features"], identities
    )
    contexts, point_values = {"coarse16": {}, "fine32": {}}, {}
    for vid in identities:
        verify_binding(coarse[vid], fine[vid], points[vid], media[vid])
        path = Path(media[vid]["path"])
        if sha(path) != media[vid]["sha256"]:
            raise ValueError("actual train media changed")
        hashes[str(path)] = media[vid]["sha256"]
        point_values[vid] = np.array(native_array(points[vid], (256, 512), np.float32, hashes))
        for arm, row, cells in [("coarse16", coarse[vid], 256), ("fine32", fine[vid], 1024)]:
            outputs = row["outputs"]
            a = native_array(outputs["appearance"], (32, cells, 384), np.float32, hashes)
            p = native_array(outputs["positions"], (32, cells, 3), np.float32, hashes)
            t = native_array(outputs["frames"], (32,), np.int64, hashes)
            np.testing.assert_array_equal(t, row["frame_indices"])
            expected_xy = np.array(
                [
                    [(x + 0.5) / int(np.sqrt(cells)), (y + 0.5) / int(np.sqrt(cells))]
                    for y in range(int(np.sqrt(cells)))
                    for x in range(int(np.sqrt(cells)))
                ],
                np.float32,
            )
            np.testing.assert_array_equal(p[..., :2], np.broadcast_to(expected_xy, (32, cells, 2)))
            np.testing.assert_array_equal(
                p[..., 2],
                np.broadcast_to(
                    (t.astype(np.float32) / max(1, row["original_frames"] - 1))[:, None],
                    (32, cells),
                ),
            )
            contexts[arm][vid] = (outputs["appearance"]["path"], outputs["positions"]["path"])
            del a, p, t  # Do not retain 5GB of mapped pages in the process working set.
    text_report = reports["perception-occlusion-text-features-01/result.json"]
    row = text_report["features"]["train"]
    text = native_array(row, (198, 3, 1536), np.float32, hashes)
    expected_rows = [
        dict(video_id=p["video_id"], condition=q["condition"], target=q["target"])
        for p in pairs
        for q in p["questions"]
    ]
    if row["rows"] != expected_rows:
        raise ValueError("original text/target identity order mismatch")
    if (
        text_report["revision"] != "4ca70771034acceecb2e72475f72050fcdde4ddc"
        or text_report["parameters"] != 17389824
    ):
        raise ValueError("frozen text revision/count mismatch")
    with np.load(ROOT / "spatial-readout-full-01/train-labels.npz", allow_pickle=False) as labels:
        cover, available, qa, ids = [
            labels[k].copy() for k in ("cover", "available", "qa", "identities")
        ]
    if (
        cover.shape != (99, 256)
        or cover.dtype != bool
        or available.shape != (99,)
        or available.dtype != bool
        or available.sum() != 98
        or ids.tolist() != identities
    ):
        raise ValueError("cached cover labels changed")
    np.testing.assert_array_equal(qa, [r["target"] for r in expected_rows])
    fit, hold = scene_split(identities, [media[v]["sha256"] for v in identities])
    data = {}
    for name, indices in [("fit", fit), ("selector", hold)]:
        d = paired_subset(pairs, text, indices)
        d.update(
            cover=cover[indices], available=available[indices], source_indices=indices.tolist()
        )
        data[name] = d
    # Source RGB/full-window identities are checked independently of label information.
    for key, rows in [("selected_rgb_sha256", fine), ("prepared_sha256", points)]:
        values = [rows[v][key] for v in identities]
        if len(set(values)) != 99:
            raise ValueError("duplicate native video content")
    split = {
        name: dict(
            video_ids=[p["video_id"] for p in d["pairs"]],
            media_sha256=[media[p["video_id"]]["sha256"] for p in d["pairs"]],
            original_questions=d["pairs"],
            source_indices=d["source_indices"],
        )
        for name, d in data.items()
    }
    return point_values, contexts, data, hashes, split
