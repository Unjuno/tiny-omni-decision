import hashlib

import numpy as np
import pytest

from tiny_omni_decision import native_proposal_corpus as module

IDS = (
    "video_10103",
    "video_10186",
    "video_10212",
    "video_10387",
    "video_1087",
    "video_10886",
    "video_10894",
    "video_10903",
)


def fixture():
    sources = [
        dict(
            video_id=i,
            split="train",
            media_path=i + ".mp4",
            media_sha256="a" * 64,
            cut_frame=100,
            frames=100,
        )
        for i in IDS
    ]
    times = {i: np.rint(np.linspace(0, 99, 32)).astype(np.int64) for i in IDS}
    return sources, times


def test_fixed_eight_train_inputs_and_all32_times_are_preserved():
    sources, times = fixture()
    plans = module.plan_native_corpus(sources, times)
    assert [p["video_id"] for p in plans] == list(IDS)
    assert all(p["frame_indices"] == times[p["video_id"]].tolist() for p in plans)
    assert sum(len(p["frame_indices"]) for p in plans) == 256
    assert "frame_indices" not in sources[0]
    assert plans == module.plan_native_corpus(sources, times)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "reordered",
        "validation",
        "wrong_cut",
        "too_short",
        "bad_sha",
        "time_missing",
        "time_extra",
        "wrong_time",
        "float_time",
    ],
)
def test_incomplete_or_changed_fixed_source_plan_is_rejected(kind):
    sources, times = fixture()
    if kind == "missing":
        sources.pop()
    elif kind == "duplicate":
        sources[-1] = sources[0].copy()
    elif kind == "reordered":
        sources.reverse()
    elif kind == "validation":
        sources[0]["split"] = "validation"
    elif kind == "wrong_cut":
        sources[0]["cut_frame"] = 90
    elif kind == "too_short":
        sources[0]["frames"] = 31
    elif kind == "bad_sha":
        sources[0]["media_sha256"] = "unknown"
    elif kind == "time_missing":
        times.pop(IDS[0])
    elif kind == "time_extra":
        times["another"] = times[IDS[0]]
    elif kind == "wrong_time":
        times[IDS[0]][1] = 2
    else:
        times[IDS[0]] = times[IDS[0]].astype(np.float32)
    with pytest.raises(ValueError):
        module.plan_native_corpus(sources, times)


def arrays():
    return dict(
        masks_packed=np.zeros((1024, 8192), np.uint8),
        scores=np.zeros(1024, np.float32),
        boxes=np.zeros((1024, 4), np.float64),
        points=np.zeros((1024, 2), np.float32),
        nms_indices=np.empty(0, np.int64),
    )


def test_all1024_arrays_admitted_without_topk_selection():
    v = arrays()
    module.validate_native_proposal_arrays(v)
    v["nms_indices"] = np.arange(1024, dtype=np.int64)
    module.validate_native_proposal_arrays(v)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "truncated_masks",
        "wrong_mask_dtype",
        "half_scores",
        "nan_score",
        "nan_point",
        "box_outside",
        "box_reversed",
        "nms_duplicate",
        "nms_outside",
        "nms_float",
    ],
)
def test_invalid_archive_cannot_become_native_corpus(kind):
    v = arrays()
    if kind == "missing":
        v.pop("scores")
    elif kind == "extra":
        v["selected_indices"] = np.array([1, 2, 3])
    elif kind == "truncated_masks":
        v["masks_packed"] = v["masks_packed"][:3]
    elif kind == "wrong_mask_dtype":
        v["masks_packed"] = v["masks_packed"].astype(np.int8)
    elif kind == "half_scores":
        v["scores"] = v["scores"].astype(np.float16)
    elif kind == "nan_score":
        v["scores"][0] = np.nan
    elif kind == "nan_point":
        v["points"][0, 0] = np.nan
    elif kind == "box_outside":
        v["boxes"][0, 2] = 1.1
    elif kind == "box_reversed":
        v["boxes"][0, :2] = 0.5
    elif kind == "nms_duplicate":
        v["nms_indices"] = np.array([1, 1], np.int64)
    elif kind == "nms_outside":
        v["nms_indices"] = np.array([1024], np.int64)
    else:
        v["nms_indices"] = np.array([1.0], np.float32)
    with pytest.raises(ValueError):
        module.validate_native_proposal_arrays(v)


def smoke_report():
    sources, times = fixture()
    plans = module.plan_native_corpus(sources, times)
    report = dict(
        status="PASS_NATIVE_CORPUS_CACHE_SMOKE",
        initial_parent_byte_exact=True,
        cached_fresh_raw_exact=True,
        rows=[
            dict(
                video_id=IDS[0],
                frame_index=t,
                raw_proposals=1024,
                encoder_calls=2,
                readback_bitwise=True,
                cache_prefix=dict(path="prefix.npz", sha256="a" * 64),
            )
            for t in plans[0]["frame_indices"][:2]
        ],
    )
    return report, plans


def test_smoke_requires_both_fixed_source_frames_and_complete_outputs():
    report, plans = smoke_report()
    module.validate_native_smoke_report(report, plans)


@pytest.mark.parametrize(
    "kind", ["missing_frame", "wrong_clip", "false_exact", "partial", "wrong_encoder", "no_prefix"]
)
def test_incomplete_or_unrelated_smoke_cannot_admit_long_corpus(kind):
    report, plans = smoke_report()
    if kind == "missing_frame":
        report["rows"].pop()
    elif kind == "wrong_clip":
        report["rows"][0]["video_id"] = IDS[1]
    elif kind == "false_exact":
        report["initial_parent_byte_exact"] = False
    elif kind == "partial":
        report["rows"][1]["raw_proposals"] = 64
    elif kind == "wrong_encoder":
        report["rows"][1]["encoder_calls"] = 3
    else:
        report["rows"][1]["cache_prefix"] = None
    with pytest.raises(ValueError):
        module.validate_native_smoke_report(report, plans)


def test_reference_config_is_bound_to_actual_bytes(tmp_path):
    p = tmp_path / 'effective-config.json'
    p.write_bytes(b'original-config')
    pin = hashlib.sha256(p.read_bytes()).hexdigest()
    assert module.require_reference_config(p, pin, pin) == pin
    p.write_bytes(b'changed-config')
    with pytest.raises(ValueError, match='reference config'):
        module.require_reference_config(p, pin, pin)


def test_reference_report_cannot_claim_another_configuration(tmp_path):
    p = tmp_path / 'effective-config.json'
    p.write_bytes(b'original-config')
    pin = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='reference config'):
        module.require_reference_config(p, 'f' * 64, pin)
