"""Separate posthoc source-box opportunity; no model or GPU inference."""

import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.probe_dynamic_discovery import OLD_PROPOSAL_PIN, R1, legacy_functions
from scripts.probe_track_appearance import ROOT, sha, write
from tiny_omni_decision.proposal_opportunity import bbox_opportunity, source_box_at_time

ANNOTATION_PIN = "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"


def validate_proposal_bank(saved):
    schema = dict(
        masks_packed=((1024, 8192), np.uint8),
        scores=((1024,), np.float32),
        boxes=((1024, 4), np.float64),
        points=((1024, 2), np.float32),
    )
    if set(saved) != set(schema) | {"nms_indices"}:
        raise ValueError("complete proposal bank keys required before annotations")
    for key, (shape, dtype) in schema.items():
        value = saved[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError("invalid complete proposal array: " + key)
    boxes = saved["boxes"]
    if (boxes < 0).any() or (boxes > 1).any() or (boxes[:, :2] > boxes[:, 2:]).any():
        raise ValueError("invalid normalized proposal boxes")
    indices = saved["nms_indices"]
    if (
        indices.ndim != 1
        or indices.dtype != np.int64
        or (indices < 0).any()
        or (indices >= 1024).any()
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("invalid NMS proposal inventory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh report path required")
    start = time.perf_counter()
    result_path = args.run / "result.json"
    inference_result_pin = sha(result_path)
    result = json.loads(result_path.read_text())
    config_path = args.run / "config.json"
    if result["status"] != "PASS_FIXED32_DYNAMIC_PROPOSALS_NOT_QA" or len(result["rows"]) != 32:
        raise ValueError("complete32-frame inference required before annotation access")
    if sha(config_path) != result["config_sha256"]:
        raise ValueError("run config changed")
    config = json.loads(config_path.read_text())
    plan = config["input_plan"]
    if (
        plan["video_id"] != "video_10903"
        or plan["split"] != "train"
        or [r["frame_index"] for r in result["rows"]] != plan["frame_indices"]
    ):
        raise ValueError("exact known train scene/time binding required")
    for path, pin in config["source_hashes"].items():
        if sha(path) != pin:
            raise ValueError("protected inference source changed")
    baseline_path = ROOT / "perception-sam2-auto-init-02/video_10903-proposals.npz"
    if sha(baseline_path) != OLD_PROPOSAL_PIN:
        raise ValueError("original proposal bank changed")
    with np.load(baseline_path, allow_pickle=False) as saved:
        baseline_boxes = saved["boxes"].copy()
        baseline_nms = saved["nms_indices"].copy()
    # Complete artifact admission is independent of source labels: verify every
    # bank before the archive containing annotations can be opened.
    for row in result["rows"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("complete proposal bank hash changed before annotations")
        with np.load(row["path"], allow_pickle=False) as saved:
            validate_proposal_bank(saved)
    archive_path = ROOT / "perception-test-source-audit-01/train_annotations.zip"
    if sha(archive_path) != ANNOTATION_PIN:
        raise ValueError("training annotation archive changed")
    # The complete raw inference bundle exists before this process opens labels.
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["all_train.json"]:
            raise ValueError("unexpected source archive")
        source = json.loads(archive.read("all_train.json"))[plan["video_id"]]
    if source["metadata"]["split"] != "train" or source["metadata"]["video_id"] != plan["video_id"]:
        raise ValueError("source identity/split changed")
    role_suffixes = {
        "group": "Track these similar or identical objects.",
        "hidden": "Track the occluded object.",
    }
    roles = {}
    for role, suffix in role_suffixes.items():
        questions = [q for q in source["grounded_question"] if q["question"].endswith(suffix)]
        if len(questions) != 1 or not questions[0]["answers"]:
            raise ValueError("exact grounded group/hidden target required; no flag replacement")
        roles[role] = questions[0]["answers"]
    if len(roles["hidden"]) != 1 or len(set(roles["group"])) != len(roles["group"]):
        raise ValueError("unique source roles required")
    identities = roles["group"] + roles["hidden"]
    tracks = {
        identity: next(t for t in source["object_tracking"] if t["id"] == identity)
        for identity in identities
    }
    helpers = legacy_functions()
    rows = []
    for row in result["rows"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("proposal bank changed")
        with np.load(row["path"], allow_pickle=False) as saved:
            packed = saved["masks_packed"]
            if packed.shape != (1024, 8192) or packed.dtype != np.uint8:
                raise ValueError("all raw proposals required")
            masks = (
                np.unpackbits(packed, axis=1, bitorder="big").reshape(1024, 256, 256).astype(bool)
            )
            boxes = helpers["boxes_from_masks"](masks)
            if boxes.tobytes() != saved["boxes"].tobytes():
                raise ValueError("independent mask-to-box check failed")
            nms = np.asarray(helpers["nms"](boxes, saved["scores"]), np.int64)
            if nms.tobytes() != saved["nms_indices"].tobytes():
                raise ValueError("NMS recomputation failed")
            for identity in identities:
                target = source_box_at_time(tracks[identity], row["frame_index"])
                rows.append(
                    dict(
                        frame_index=row["frame_index"],
                        source_id=identity,
                        role="hidden" if identity in roles["hidden"] else "group",
                        source_label=tracks[identity]["label"],
                        source_proxy=target,
                        dynamic=bbox_opportunity(boxes, nms, target["box"]),
                        first_frame_static=bbox_opportunity(
                            baseline_boxes, baseline_nms, target["box"]
                        ),
                    )
                )
    summary = {}
    for role in ("group", "hidden"):
        selected = [r for r in rows if r["role"] == role]
        available = [r for r in selected if r["source_proxy"]["box"] is not None]
        summary[role] = dict(
            source_ids=roles[role],
            all_source_time_cases=len(selected),
            source_box_available_cases=len(available),
            source_box_unavailable_cases=len(selected) - len(available),
            exact_annotation_time_cases=sum(
                r["source_proxy"]["source_frame_exact"] for r in available
            ),
            max_nearest_annotation_distance_frames=max(
                r["source_proxy"]["nearest_annotation_distance_frames"] for r in available
            )
            if available
            else None,
        )
        for policy in ("dynamic", "first_frame_static"):
            summary[role][policy] = {}
            for stage in ("raw", "nms"):
                hits = sum(r[policy][stage + "_max_iou"] >= 0.5 for r in available)
                summary[role][policy][stage] = dict(
                    opportunity_hits=hits,
                    denominator=len(available),
                    bbox_opportunity_rate=hits / len(available) if available else None,
                )
    for path, pin in config["source_hashes"].items():
        if sha(path) != pin:
            raise ValueError("protected input changed while scoring")
    if sha(result_path) != inference_result_pin or sha(archive_path) != ANNOTATION_PIN:
        raise ValueError("result/annotation changed")
    report = dict(
        status="PASS_POSTHOC_TEMPORAL_BBOX_OPPORTUNITY_NOT_QA",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        scorer_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path("src/tiny_omni_decision/proposal_opportunity.py")),
        inference_result_sha256=inference_result_pin,
        inference_config_sha256=sha(config_path),
        annotation_sha256=ANNOTATION_PIN,
        baseline_proposals_sha256=OLD_PROPOSAL_PIN,
        geometry_helper_source_sha256=sha(R1 / "auto_init02.py"),
        rows=rows,
        summary=summary,
        threshold=0.5,
        source_boxes_interpolated_without_extrapolation=True,
        model_inference_after_annotation_load=False,
        training=False,
        validation_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        seconds=time.perf_counter() - start,
        limitation="Known train regression scene; bbox opportunity is not identity or QA quality",
    )
    write(args.output, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
