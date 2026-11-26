"""Fixed train-only native proposal corpus admission; no source labels or model."""

import hashlib
import re
from pathlib import Path

import numpy as np

FIXED_IDS = (
    "video_10103",
    "video_10186",
    "video_10212",
    "video_10387",
    "video_1087",
    "video_10886",
    "video_10894",
    "video_10903",
)
PROPOSAL_KEYS = ("masks_packed", "scores", "boxes", "points", "nms_indices")


def require_reference_config(path, reported_hash, expected_hash):
    """A report field alone cannot prove an unchanged actual reference config."""
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != reported_hash or actual != expected_hash:
        raise ValueError("actual reference config must match both report and current contract")
    return actual


def plan_native_corpus(sources, time_banks):
    """Keep eight previously pinned train clips and their32 uniform source times."""
    if [p["video_id"] for p in sources] != list(FIXED_IDS) or set(time_banks) != set(FIXED_IDS):
        raise ValueError("complete ordered fixed eight training clips required")
    plans = []
    for source in sources:
        times = time_banks[source["video_id"]]
        frames = source["frames"]
        if (
            source["split"] != "train"
            or type(frames) is not int
            or frames < 32
            or source["cut_frame"] not in (-1, frames)
            or not re.fullmatch("[0-9a-f]{64}", source["media_sha256"])
            or not isinstance(times, np.ndarray)
            or times.shape != (32,)
            or times.dtype.kind not in "iu"
            or not np.array_equal(times, np.rint(np.linspace(0, frames - 1, 32)))
        ):
            raise ValueError("pinned native train media and every32 source time required")
        plans.append(
            {
                **{
                    k: source[k]
                    for k in (
                        "video_id",
                        "split",
                        "media_path",
                        "media_sha256",
                        "cut_frame",
                        "frames",
                    )
                },
                "frame_indices": times.tolist(),
            }
        )
    return plans


def validate_native_proposal_arrays(values):
    """Require the full1024 proposal inventory; no legacy selected/top-k field."""
    schema = dict(
        masks_packed=((1024, 8192), np.dtype("uint8")),
        scores=((1024,), np.dtype("float32")),
        boxes=((1024, 4), np.dtype("float64")),
        points=((1024, 2), np.dtype("float32")),
    )
    if set(values) != set(PROPOSAL_KEYS):
        raise ValueError("exact five full native proposal arrays required")
    for key, (shape, dtype) in schema.items():
        value = values[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("full finite native proposal schema required: " + key)
    boxes, ids = values["boxes"], values["nms_indices"]
    if (
        np.any(boxes < 0)
        or np.any(boxes > 1)
        or np.any(boxes[:, :2] > boxes[:, 2:])
        or not isinstance(ids, np.ndarray)
        or ids.dtype != np.int64
        or ids.ndim != 1
        or np.any(ids < 0)
        or np.any(ids >= 1024)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("normalized boxes and unique full-range native NMS identities required")


def validate_native_smoke_report(report, plans):
    """Require both fixed first-source frames, with cached/fresh and full output proof."""
    rows = report["rows"]
    expected = [(plans[0]["video_id"], t) for t in plans[0]["frame_indices"][:2]]
    if (
        report["status"] != "PASS_NATIVE_CORPUS_CACHE_SMOKE"
        or report.get("initial_parent_byte_exact") is not True
        or report.get("cached_fresh_raw_exact") is not True
        or [(r["video_id"], r["frame_index"]) for r in rows] != expected
        or any(
            r["raw_proposals"] != 1024
            or r["encoder_calls"] != 2
            or r["readback_bitwise"] is not True
            or not r.get("cache_prefix")
            for r in rows
        )
    ):
        raise ValueError("complete fixed two-frame native cache smoke required")
