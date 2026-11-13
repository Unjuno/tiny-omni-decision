"""Extract label-free candidate-frame ROI features with the pinned VideoPrism model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tiny_omni_decision.perception_video_preprocessing import letterbox_videoprism_frames
from tiny_omni_decision.perception_video_roi import roi_average
from tiny_omni_decision.video_token_grid import token_grid

SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_PARAMETERS = 114_365_184


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_plan_split(plan, media_report, split_name, media_report_sha256):
    expected_format = (
        "perception_test_source_candidate_validation_feature_plan_v1"
        if split_name == "validation"
        else "perception_test_source_candidate_feature_plan_v1"
    )
    if plan.get("format") != expected_format:
        raise ValueError("unexpected source-candidate feature plan format")
    if not plan.get("label_free") or plan.get("model_loaded") or plan.get("metrics_computed"):
        raise ValueError("feature plan is not a frozen label-free preparation artifact")
    report_key = (
        "validation_media_report_sha256"
        if split_name == "validation"
        else f"{split_name}_media_report_sha256"
    )
    if plan.get(report_key) != media_report_sha256:
        raise ValueError(f"{split_name} media report hash differs from frozen plan")
    report_rows = {row["record_id"]: row for row in media_report["records"]}
    records = plan[split_name]["records"]
    if len(report_rows) != len(media_report["records"]):
        raise ValueError("media report contains duplicate record IDs")
    if set(report_rows) != {row["record_id"] for row in records}:
        raise ValueError(f"{split_name} media report IDs differ from feature plan")
    for row in records:
        report_row = report_rows[row["record_id"]]
        if (
            report_row.get("video_id") != row["video_id"]
            or report_row.get("media_sha256") is None
            or row.get("terminal_frame_repair") != report_row.get("terminal_frame_repair")
        ):
            raise ValueError(f"media report row differs from feature plan: {row['record_id']}")
        available = set(report_row["decoded_frame_ids"])
        for candidate in row["candidates"]:
            for frame in candidate["frames"]:
                if frame["frame_id"] not in available:
                    raise ValueError(
                        f"feature plan requests a frame absent from preflight: {row['record_id']}"
                    )
    return report_rows


def decode_video_frames(path: Path, wanted_frame_ids: set[int]) -> dict[int, np.ndarray]:
    import av

    if not wanted_frame_ids:
        raise ValueError("cannot decode an empty frame set")
    decoded = {}
    max_frame = max(wanted_frame_ids)
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        for index, frame in enumerate(container.decode(streams[0])):
            if index in wanted_frame_ids:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if index >= max_frame:
                break
    missing = sorted(wanted_frame_ids - decoded.keys())
    if missing:
        raise ValueError(f"preflighted video frames became unavailable: {path.name} {missing}")
    return decoded


def _load_videoprism(checkpoint: Path, source: Path):
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("VideoPrism source checkout differs from pinned pristine revision")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax
    import jax.numpy as jnp
    import tensorflow as tf

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("candidate VideoPrism extraction is pinned to CPU backend")
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise ValueError("TensorFlow GPU was initialized before CPU isolation") from error
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0, str(source))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameters = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(variables))
    if parameters != MODEL_PARAMETERS:
        raise ValueError(f"unexpected VideoPrism parameter count: {parameters}")
    encode = jax.jit(lambda batch: model.apply(variables, batch, train=False))
    return jax, jnp, tf, variables, encode, parameters, commit


def run(args: argparse.Namespace) -> dict:
    plan_path = args.plan.resolve(strict=True)
    acquisition_path = args.acquisition_manifest.resolve(strict=True)
    media_report_path = args.media_report.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source = args.source.resolve(strict=True)
    media_dir = args.media_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    media_report = json.loads(media_report_path.read_text(encoding="utf-8"))
    report_hash = sha256_file(media_report_path)
    split_name = args.split
    if media_report.get("status") not in {"PASS", "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR"}:
        raise ValueError("media report did not pass the CPU preflight")
    if media_report.get("acquisition_manifest_sha256") != sha256_file(acquisition_path):
        raise ValueError("media report acquisition-manifest hash mismatch")
    report_rows = _validate_plan_split(plan, media_report, split_name, report_hash)
    if plan.get("acquisition_manifest_sha256") != sha256_file(acquisition_path):
        raise ValueError("feature plan acquisition-manifest hash mismatch")
    if args.limit_videos is not None and args.limit_videos < 1:
        raise ValueError("limit-videos must be positive")

    clips = {}
    for clip in acquisition["clips"]:
        video_id = clip["metadata"]["video_id"]
        if video_id in clips:
            raise ValueError(f"duplicate media receipt for {video_id}")
        clips[video_id] = clip
    split_plan = plan[split_name]
    video_plans = split_plan["videos"]
    if args.limit_videos is not None:
        video_plans = video_plans[: args.limit_videos]
    selected_video_ids = {row["video_id"] for row in video_plans}
    split_records = [row for row in split_plan["records"] if row["video_id"] in selected_video_ids]
    records_by_video: dict[str, list[dict]] = {}
    for row in split_records:
        records_by_video.setdefault(row["video_id"], []).append(row)
    media_hash_by_video = {}
    for report_row in report_rows.values():
        previous = media_hash_by_video.setdefault(
            report_row["video_id"], report_row["media_sha256"]
        )
        if previous != report_row["media_sha256"]:
            raise ValueError("one video maps to multiple preflight media hashes")
    for video_plan in video_plans:
        if media_hash_by_video.get(video_plan["video_id"]) != video_plan["media_sha256"]:
            raise ValueError(
                f"feature plan media hash differs from preflight: {video_plan['video_id']}"
            )

    jax, jnp, tf, variables, encode, parameters, commit = _load_videoprism(checkpoint, source)
    from tiny_omni_decision.perception_video_windows import plan_frame_windows

    output.mkdir(parents=True)
    run_metadata = {
        "status": "RUNNING",
        "plan_sha256": sha256_file(plan_path),
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "media_report_sha256": report_hash,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": commit,
        "model_parameters": parameters,
        "device": "CPU",
        "split": split_name,
        "selected_video_count": len(video_plans),
        "expected_split_video_count": split_plan["video_group_count"],
        "complete_split": args.limit_videos is None,
        "input_frames": 8,
        "preprocessing": "bilinear square letterbox to 288, black pad, RGB / 255",
        "feature": "candidate-specific area-weighted VideoPrism spatial ROI mean per frame",
        "labels_loaded": False,
        "gpu_used": False,
        "metrics_computed": False,
    }
    atomic_json(output / "run.json", run_metadata)

    def resize_bilinear(frame, size):
        return tf.image.resize(frame, size, method="bilinear").numpy()

    vector_rows = []
    vector_index_rows = []
    started = time.perf_counter()
    video_count = len(video_plans)
    for video_index, video_plan in enumerate(video_plans, start=1):
        video_id = video_plan["video_id"]
        clip = clips.get(video_id)
        if clip is None or clip["sha256"] != video_plan["media_sha256"]:
            raise ValueError(f"feature plan media hash differs from acquisition: {video_id}")
        path = media_dir / f"{video_id}.mp4"
        if path.stat().st_size != clip["bytes"] or sha256_file(path) != clip["sha256"]:
            raise ValueError(f"media file differs from fixed acquisition receipt: {video_id}")
        wanted_frames = set(video_plan["frame_ids"])
        frames_by_id = decode_video_frames(path, wanted_frames)
        references = {}
        for record in records_by_video.get(video_id, []):
            for candidate in record["candidates"]:
                for frame in candidate["frames"]:
                    window_index, slot_index = frame["window_slot"]
                    references.setdefault(window_index, {}).setdefault(slot_index, []).append(
                        {
                            "record_id": record["record_id"],
                            "candidate_id": candidate["candidate_id"],
                            "frame_id": frame["frame_id"],
                            "roi_xyxy_tokens": frame["roi_xyxy_tokens"],
                        }
                    )
        expected_window_map = {
            str(frame_id): list(slot)
            for frame_id, slot in plan_frame_windows(sorted(wanted_frames))[
                "frame_to_window_slot"
            ].items()
        }
        if expected_window_map != video_plan["frame_to_window_slot"]:
            raise ValueError(f"frame-window mapping changed for {video_id}")

        for window_index, frame_ids in enumerate(video_plan["windows"]):
            raw_frames = np.stack([frames_by_id[frame_id] for frame_id in frame_ids])
            processed = letterbox_videoprism_frames(
                raw_frames, resize_bilinear, frame_size=288, pad_value=0.0
            )
            encoded = encode(jnp.asarray(processed[None], dtype=jnp.float32))
            jax.tree_util.tree_map(
                lambda value: (
                    value.block_until_ready() if hasattr(value, "block_until_ready") else value
                ),
                encoded,
            )
            tokens = encoded[0] if isinstance(encoded, tuple) else encoded
            grid = token_grid(np.asarray(tokens))
            for slot_index, refs in references.get(window_index, {}).items():
                for ref in refs:
                    if frame_ids[slot_index] != ref["frame_id"]:
                        raise ValueError(
                            "frame-window slot no longer refers to the planned source frame"
                        )
                    vector = roi_average(grid[slot_index], ref["roi_xyxy_tokens"])
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise ValueError(f"invalid candidate ROI vector: {ref['record_id']}")
                    vector = vector.astype(np.float32, copy=False)
                    feature_index = len(vector_rows)
                    vector_rows.append(vector)
                    vector_index_rows.append(
                        {
                            "feature_index": feature_index,
                            "record_id": ref["record_id"],
                            "video_id": video_id,
                            "candidate_id": ref["candidate_id"],
                            "frame_id": ref["frame_id"],
                            "feature_dim": 768,
                        }
                    )
        print(
            json.dumps(
                {
                    "completed_videos": video_index,
                    "total_videos": video_count,
                    "roi_vectors": len(vector_rows),
                    "elapsed_seconds": time.perf_counter() - started,
                    "latest_video_id": video_id,
                }
            ),
            flush=True,
        )

    if not vector_rows:
        raise ValueError("no candidate-frame ROI vectors were produced")
    if len(vector_index_rows) != sum(
        len(candidate["frames"]) for row in split_records for candidate in row["candidates"]
    ):
        raise ValueError("candidate ROI feature count does not match the label-free plan")
    features = np.stack(vector_rows)
    feature_path = output / "candidate_frame_roi_features.npy"
    temporary_feature_path = feature_path.with_suffix(".npy.tmp")
    with temporary_feature_path.open("xb") as stream:
        np.save(stream, features, allow_pickle=False)
    temporary_feature_path.replace(feature_path)
    index_path = output / "feature_index.jsonl"
    with index_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in vector_index_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    result = {
        "status": (
            "PASS_VIDEO_FEATURE_SMOKE"
            if args.limit_videos is not None
            else "PASS_SOURCE_CANDIDATE_VALIDATION_VIDEO_FEATURES"
            if split_name == "validation"
            else "PASS_SOURCE_CANDIDATE_VIDEO_FEATURES"
        ),
        "complete_split": args.limit_videos is None,
        "split": split_name,
        "plan_sha256": sha256_file(plan_path),
        "media_report_sha256": report_hash,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": commit,
        "model_parameters": parameters,
        "device": "CPU",
        "video_count": video_count,
        "record_count": len(split_records),
        "roi_vector_count": len(vector_index_rows),
        "feature_shape": list(features.shape),
        "feature_sha256": sha256_file(feature_path),
        "feature_index_sha256": sha256_file(index_path),
        "elapsed_seconds": time.perf_counter() - started,
        "labels_loaded": False,
        "gpu_used": False,
        "metrics_computed": False,
        "validation_scope": (
            "fixed official validation transfer; non-blind"
            if split_name == "validation"
            else "internal fixed development partition; non-blind"
        ),
    }
    atomic_json(output / "result.json", result)
    atomic_json(
        output / "run.json",
        {**run_metadata, **result, "result_sha256": sha256_file(output / "result.json")},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--media-report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "validation"), required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-videos", type=int)
    print(json.dumps(run(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
