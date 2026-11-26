import numpy as np
import pytest

from tiny_omni_decision import native_proposal_audit as module


def values():
    masks = np.zeros((1024, 256, 256), np.uint8)
    masks[0, 32:96, 64:128] = 1
    masks[1] = masks[0]
    masks[2, 128:192, 128:192] = 1
    points = np.array(
        [[(x + 0.5) * 1920 / 32, (y + 0.5) * 1080 / 32] for y in range(32) for x in range(32)],
        np.float32,
    )
    boxes = np.zeros((1024, 4), np.float64)
    boxes[0] = boxes[1] = [0.25, 0.125, 0.5, 0.375]
    boxes[2] = [0.5, 0.5, 0.75, 0.75]
    scores = np.ones(1024, np.float32)
    return dict(
        masks_packed=np.packbits(masks.reshape(1024, -1), axis=1, bitorder="big"),
        scores=scores,
        boxes=boxes,
        points=points,
        nms_indices=np.array([0, 2], np.int64),
    )


def test_independent_geometry_grid_and_stable_tied_nms_reconstruction():
    v = values()
    before = {k: a.tobytes() for k, a in v.items()}
    result = module.audit_native_arrays(v, height=1080, width=1920)
    assert result == dict(
        raw_proposals=1024,
        nonempty_masks=3,
        positive_presence=1024,
        nms_count=2,
        nms_reconstructed_exact=True,
        boxes_reconstructed_exact=True,
        points_reconstructed_exact=True,
    )
    assert {k: a.tobytes() for k, a in v.items()} == before


def test_all_empty_inventory_remains_valid_and_keeps_full_proposal_count():
    v = values()
    v["masks_packed"][:] = 0
    v["boxes"][:] = 0
    v["scores"][:] = -1
    v["nms_indices"] = np.empty(0, np.int64)
    result = module.audit_native_arrays(v, height=1080, width=1920)
    assert result["raw_proposals"] == 1024 and result["nonempty_masks"] == result["nms_count"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "box",
        "nms_order",
        "nms_missing",
        "nms_duplicate",
        "point_order",
        "point_width",
        "negative_presence",
        "score_nan",
        "extra_key",
        "wrong_dtype",
        "truncated",
    ],
)
def test_storage_that_disagrees_with_native_geometry_cannot_pass(kind):
    v = values()
    if kind == "box":
        v["boxes"][0, 0] += 1 / 256
    elif kind == "nms_order":
        v["nms_indices"] = np.array([2, 0], np.int64)
    elif kind == "nms_missing":
        v["nms_indices"] = np.array([0], np.int64)
    elif kind == "nms_duplicate":
        v["nms_indices"] = np.array([0, 2, 0], np.int64)
    elif kind == "point_order":
        v["points"][[0, 1]] = v["points"][[1, 0]]
    elif kind == "point_width":
        v["points"] *= 2
    elif kind == "negative_presence":
        v["scores"][0] = 0
    elif kind == "score_nan":
        v["scores"][0] = np.nan
    elif kind == "extra_key":
        v["selected_indices"] = np.array([0, 2], np.int64)
    elif kind == "wrong_dtype":
        v["scores"] = v["scores"].astype(np.float64)
    else:
        v["masks_packed"] = v["masks_packed"][:3]
    with pytest.raises(ValueError):
        module.audit_native_arrays(v, height=1080, width=1920)


def test_presence_score_order_reconstructs_different_correct_nms_identity():
    v = values()
    v["scores"][1] = 2
    v["nms_indices"] = np.array([1, 2], np.int64)
    assert module.audit_native_arrays(v, height=1080, width=1920)["nms_count"] == 2


def coverage(snapshot=False, count=256):
    cfg = dict(
        plans=[
            dict(video_id=v, frame_indices=list(range(32)))
            for v in (
                "video_10103",
                "video_10186",
                "video_10212",
                "video_10387",
                "video_1087",
                "video_10886",
                "video_10894",
                "video_10903",
            )
        ]
    )
    rows = [
        dict(video_id=p["video_id"], frame_index=t)
        for p in cfg["plans"]
        for t in p["frame_indices"]
    ][:count]
    report = dict(
        status="RUNNING" if snapshot else "PASS_NATIVE_EIGHT_CLIP_PROPOSAL_CORPUS", rows=rows
    )
    return report, cfg


def test_full_and_snapshot_coverage_have_distinct_admission():
    r, c = coverage()
    assert module.audit_report_coverage(r, c, snapshot=False) == 256
    r, c = coverage(snapshot=True, count=15)
    assert module.audit_report_coverage(r, c, snapshot=True) == 15


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "partial_final",
        "wrong_order",
        "gap",
        "float_time",
        "duplicate_extra",
        "running_final",
    ],
)
def test_incomplete_or_reordered_reports_cannot_claim_full_admission(kind):
    r, c = coverage()
    if kind == "empty":
        r["rows"] = []
    elif kind == "partial_final":
        r["rows"] = r["rows"][:15]
    elif kind == "wrong_order":
        r["rows"][0], r["rows"][1] = r["rows"][1], r["rows"][0]
    elif kind == "gap":
        r["rows"].pop(15)
    elif kind == "float_time":
        r["rows"][0]["frame_index"] = 0.0
    elif kind == "duplicate_extra":
        r["rows"].append(r["rows"][-1])
    else:
        r["status"] = "RUNNING"
    with pytest.raises(ValueError):
        module.audit_report_coverage(r, c, snapshot=False)


def test_snapshot_never_accepts_terminal_status_as_current_live_evidence():
    r, c = coverage()
    with pytest.raises(ValueError):
        module.audit_report_coverage(r, c, snapshot=True)
