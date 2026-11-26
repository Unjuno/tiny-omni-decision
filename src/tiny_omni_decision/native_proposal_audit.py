"""Independent packed-mask geometry/grid audit; no extractor or model imports."""

import numpy as np


def audit_report_coverage(report, config, *, snapshot):
    """A saved prefix is never a complete corpus or proof that a PID is live."""
    expected = [(p["video_id"], t) for p in config["plans"] for t in p["frame_indices"]]
    observed = [(r["video_id"], r["frame_index"]) for r in report["rows"]]
    status = "RUNNING" if snapshot else "PASS_NATIVE_EIGHT_CLIP_PROPOSAL_CORPUS"
    if (
        report["status"] != status
        or len(expected) != 256
        or not observed
        or observed != expected[: len(observed)]
        or any(type(t) is not int for _, t in observed)
        or (not snapshot and observed != expected)
    ):
        raise ValueError("ordered native coverage and correct snapshot/terminal status required")
    return len(observed)


def audit_native_arrays(values, *, height, width):
    """Reconstruct every box and full score-ordered NMS from saved binary masks."""
    shapes = dict(
        masks_packed=((1024, 8192), np.dtype("uint8")),
        scores=((1024,), np.dtype("float32")),
        boxes=((1024, 4), np.dtype("float64")),
        points=((1024, 2), np.dtype("float32")),
    )
    if (
        type(height) is not int
        or type(width) is not int
        or min(height, width) < 1
        or set(values) != {*shapes, "nms_indices"}
    ):
        raise ValueError("native dimensions and full five-array inventory required")
    for key, (shape, dtype) in shapes.items():
        a = values[key]
        if (
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("invalid native storage: " + key)
    ids = values["nms_indices"]
    if not isinstance(ids, np.ndarray) or ids.dtype != np.int64 or ids.ndim != 1:
        raise ValueError("integer native NMS sequence required")
    masks = np.unpackbits(values["masks_packed"], axis=1, bitorder="big").reshape(1024, 256, 256)
    row_present, column_present = masks.any(2), masks.any(1)
    nonempty = row_present.any(1)
    boxes = (
        np.column_stack(
            (
                column_present.argmax(1),
                row_present.argmax(1),
                256 - column_present[:, ::-1].argmax(1),
                256 - row_present[:, ::-1].argmax(1),
            )
        ).astype(np.float64)
        / 256
    )
    boxes[~nonempty] = 0
    if boxes.tobytes() != values["boxes"].tobytes():
        raise ValueError("packed masks do not reconstruct declared boxes exactly")
    if np.any(nonempty & (values["scores"] <= 0)):
        raise ValueError("native no-object presence must not retain foreground pixels")
    x = (np.arange(32, dtype=np.float64) + 0.5) * width / 32
    y = (np.arange(32, dtype=np.float64) + 0.5) * height / 32
    points = np.column_stack((np.tile(x, 32), np.repeat(y, 32))).astype(np.float32)
    if points.tobytes() != values["points"].tobytes():
        raise ValueError("native grid points/order disagree with source dimensions")
    remaining = sorted(
        np.flatnonzero(nonempty).tolist(), key=lambda i: (-float(values["scores"][i]), i)
    )
    kept = []
    while remaining:
        i = remaining.pop(0)
        kept.append(i)
        if not remaining:
            break
        other = boxes[remaining]
        intersection = np.maximum(
            np.minimum(boxes[i, 2:], other[:, 2:]) - np.maximum(boxes[i, :2], other[:, :2]), 0
        ).prod(1)
        total_area = (boxes[i, 2:] - boxes[i, :2]).prod() + (other[:, 2:] - other[:, :2]).prod(1)
        union = total_area - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        remaining = [index for index, ratio in zip(remaining, overlap, strict=True) if ratio <= 0.7]
    if ids.tobytes() != np.asarray(kept, np.int64).tobytes():
        raise ValueError("score-ordered full NMS does not reconstruct declared identities")
    return dict(
        raw_proposals=1024,
        nonempty_masks=int(nonempty.sum()),
        positive_presence=int((values["scores"] > 0).sum()),
        nms_count=len(kept),
        nms_reconstructed_exact=True,
        boxes_reconstructed_exact=True,
        points_reconstructed_exact=True,
    )
