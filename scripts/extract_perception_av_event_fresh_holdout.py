"""Extract pinned frozen DINOv2/CLAP features for one fixed TRAIN holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import load_annotations, summarize_split
from scripts.audit_perception_av_event_vision_smoke import (
    normalized_box_to_pixels,
    track_box_at_frame,
)
from scripts.extract_perception_av_event_features import (
    _decode_resampled_audio,
    _decode_selected_rgb,
    _encode_audio_event,
    _encode_visual_event,
    _load_feature_models,
    atomic_save_npy,
    atomic_write_json,
    target_index_for_candidate_order,
)
from scripts.perception_av_event_protocol import deterministic_candidate_order
from scripts.plan_perception_av_event_fresh_holdout import ANNOTATION_SHA256


def extract(
    *,
    plan_path: Path,
    plan_sha256: str,
    media_dir: Path,
    pts_path: Path,
    source_dir: Path,
    vision_dir: Path,
    clap_root: Path,
    output_dir: Path,
    device: str = "cuda",
    image_batch_size: int = 16,
) -> dict:
    import torch
    from PIL import Image

    if output_dir.exists():
        raise ValueError("fresh feature output directory required")
    if image_batch_size < 1:
        raise ValueError("image batch size must be positive")
    if sha256_file(plan_path) != plan_sha256:
        raise ValueError("holdout plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = media_dir / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    pts = json.loads(pts_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS_ACQUIRED_UNVERIFIED_DECODE"
        or manifest.get("plan_sha256") != plan_sha256
        or len(manifest.get("clips", [])) != 200
        or pts.get("status") != "PASS_COMPLETE_FRESH_HOLDOUT_PTS_AUDIT"
        or pts.get("plan_sha256") != plan_sha256
        or pts.get("media_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("complete acquisition/PTS receipts are not pinned to this plan")
    source_data, cuts, source_hashes = load_annotations(
        source_dir / "train_annotations.zip",
        source_dir / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": ANNOTATION_SHA256["train_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_train.json"],
        },
    )
    planned = {row["video_id"]: row for row in plan["rows"]}
    clips = {row["video_id"]: row for row in manifest["clips"]}
    if planned.keys() != clips.keys():
        raise ValueError("acquisition inventory differs from pinned plan")
    pts_by_event = {}
    for event in pts["events"]:
        key = (event["video_id"], event["sound_id"])
        if key in pts_by_event:
            raise ValueError("duplicate event in PTS receipt")
        pts_by_event[key] = event
    events_by_video = defaultdict(list)
    for event in summarize_split(source_data, cuts, "train")["candidate_records"]:
        if event["video_id"] in planned:
            events_by_video[event["video_id"]].append(event)
    if set(events_by_video) != set(planned) or len(pts_by_event) != plan["candidate_events"]:
        raise ValueError("annotation/PTS candidate inventory differs from fixed plan")

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; feature extraction did not begin")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    image_processor, vision, audio_frontend, audio_model = _load_feature_models(
        vision_dir, clap_root, device
    )
    features = []
    for video_index, video_id in enumerate(sorted(planned), start=1):
        clip = clips[video_id]
        media_path = Path(clip["path"])
        if not media_path.is_file() or sha256_file(media_path) != clip.get("sha256"):
            raise ValueError(f"holdout media missing or hash changed: {video_id}")
        events = sorted(events_by_video[video_id], key=lambda row: row["sound_id"])
        frame_ids = {event["selected_frame"] for event in events}
        rgb_by_frame = _decode_selected_rgb(media_path, frame_ids)
        waveform, resampled_start = _decode_resampled_audio(media_path)
        tracks = {track["id"]: track for track in source_data[video_id]["object_tracking"]}
        for event in events:
            event_key = (video_id, event["sound_id"])
            audit_row = pts_by_event.get(event_key)
            if audit_row is None:
                raise ValueError(f"PTS receipt lacks {video_id}/{event['sound_id']}")
            ordered_ids = deterministic_candidate_order(
                video_id, event["sound_id"], event["candidate_track_ids"]
            )
            source_ids = event["source_parent_ids"]
            if len(source_ids) != 1 or source_ids[0] not in ordered_ids:
                raise ValueError("event must have exactly one source among candidate tracks")
            target = target_index_for_candidate_order(ordered_ids, source_ids[0])
            frame = Image.fromarray(rgb_by_frame[event["selected_frame"]])
            crops = []
            for track_id in ordered_ids:
                box = track_box_at_frame(tracks[track_id], event["selected_frame"])
                pixel_box = normalized_box_to_pixels(box, frame.height, frame.width)
                crops.append(frame.crop(pixel_box))
            visual = _encode_visual_event(
                crops, image_processor, vision, device, image_batch_size
            )
            audio, sample_start, sample_end = _encode_audio_event(
                waveform, resampled_start, audit_row, audio_frontend, audio_model, device
            )
            stem = f"{video_id}-sound-{event['sound_id']}"
            audio_path = output_dir / f"{stem}-audio.npy"
            visual_path = output_dir / f"{stem}-visual.npy"
            atomic_save_npy(audio_path, audio)
            atomic_save_npy(visual_path, visual)
            features.append(
                {
                    "id": f"fresh_holdout:{video_id}:{event['sound_id']}",
                    "video_id": video_id,
                    "sound_id": event["sound_id"],
                    "split": "fresh_holdout",
                    "target_index": target,
                    "candidate_count": len(ordered_ids),
                    "candidate_order_sha256": hashlib.sha256(
                        ",".join(map(str, ordered_ids)).encode()
                    ).hexdigest(),
                    "media_sha256": clip["sha256"],
                    "annotation_sha256": source_hashes["annotations"],
                    "audio_sample_start": sample_start,
                    "audio_sample_end_exclusive": sample_end,
                    "audio_feature_path": str(audio_path.resolve()),
                    "audio_feature_sha256": sha256_file(audio_path),
                    "visual_feature_path": str(visual_path.resolve()),
                    "visual_feature_sha256": sha256_file(visual_path),
                }
            )
        if video_index % 10 == 0:
            atomic_write_json(
                output_dir / "progress.json",
                {"status": "RUNNING", "videos": video_index, "events": len(features)},
            )
            print(f"encoded {video_index}/200 videos; {len(features)} events", flush=True)
    if len(features) != plan["candidate_events"]:
        raise ValueError("feature event count differs from fixed candidate inventory")
    feature_path = output_dir / "features.jsonl"
    with feature_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in features:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "status": "PASS_FROZEN_FRESH_HOLDOUT_FEATURES",
        "plan_sha256": plan_sha256,
        "media_manifest_sha256": manifest_sha256,
        "pts_receipt_sha256": sha256_file(pts_path),
        "source_annotation_sha256": source_hashes,
        "feature_manifest": str(feature_path.resolve()),
        "feature_manifest_sha256": sha256_file(feature_path),
        "event_count": len(features),
        "video_count": len(planned),
        "vision_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "audio_revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        "vision_parameters_frozen": sum(p.numel() for p in vision.parameters()),
        "audio_parameters_frozen": sum(p.numel() for p in audio_model.parameters()),
        "device": device,
        "dtype": "float32",
        "image_batch_size": image_batch_size,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pyav_version": __import__("av").__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "gpu_used": device == "cuda",
        "training": False,
        "checkpoint_selection": False,
        "test_opened": False,
    }
    atomic_write_json(output_dir / "result.json", result)
    atomic_write_json(output_dir / "progress.json", {"status": "COMPLETE", "events": len(features)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--pts", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01"),
    )
    parser.add_argument(
        "--vision-dir",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/dinov2-small-audit-01/model"),
    )
    parser.add_argument(
        "--clap-root",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/clap-audit-01"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=16)
    args = parser.parse_args()
    result = extract(
        plan_path=args.plan,
        plan_sha256=args.plan_sha256,
        media_dir=args.media,
        pts_path=args.pts,
        source_dir=args.source,
        vision_dir=args.vision_dir,
        clap_root=args.clap_root,
        output_dir=args.output,
        device=args.device,
        image_batch_size=args.image_batch_size,
    )
    summary = {
        key: result[key]
        for key in (
            "status",
            "video_count",
            "event_count",
            "elapsed_seconds",
            "peak_cuda_memory_bytes",
        )
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
