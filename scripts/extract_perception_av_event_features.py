"""Extract frozen DINOv2/CLAP features for the pinned AV event protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_audio_smoke import (
    CONFIG_SHA256 as CLAP_CONFIG_SHA256,
)
from scripts.audit_perception_av_event_audio_smoke import (
    FRONTEND_SHA256 as CLAP_FRONTEND_SHA256,
)
from scripts.audit_perception_av_event_audio_smoke import (
    MAX_SAMPLES,
    TARGET_RATE,
    event_sample_bounds,
)
from scripts.audit_perception_av_event_audio_smoke import (
    WEIGHT_SHA256 as CLAP_WEIGHT_SHA256,
)
from scripts.audit_perception_av_event_media_pts import PLAN_SHA256
from scripts.audit_perception_av_event_task import (
    TRAIN_SOURCE_SHA256,
    VALID_SOURCE_SHA256,
    load_annotations,
    summarize_split,
)
from scripts.audit_perception_av_event_vision_smoke import (
    CONFIG_SHA256 as DINO_CONFIG_SHA256,
)
from scripts.audit_perception_av_event_vision_smoke import (
    PROCESSOR_SHA256 as DINO_PROCESSOR_SHA256,
)
from scripts.audit_perception_av_event_vision_smoke import (
    VISION_WEIGHT_SHA256,
    normalized_box_to_pixels,
    track_box_at_frame,
)
from scripts.clap_full_clip import duration_weighted_feature, window_spans
from scripts.perception_av_event_protocol import (
    deterministic_candidate_order,
    model_selection_video_ids,
)


def target_index_for_candidate_order(ordered_candidate_ids, source_parent_id):
    """Find the one source option after the deterministic candidate shuffle."""
    if ordered_candidate_ids.count(source_parent_id) != 1:
        raise ValueError("source parent must occur exactly once in candidate order")
    return ordered_candidate_ids.index(source_parent_id)


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_feature_models(vision_dir: Path, clap_root: Path, device: str):
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        ClapAudioConfig,
        ClapAudioModelWithProjection,
        ClapFeatureExtractor,
    )

    audio_dir = clap_root / "audio-runtime-01"
    expected = {
        vision_dir / "model.safetensors": VISION_WEIGHT_SHA256,
        vision_dir / "preprocessor_config.json": DINO_PROCESSOR_SHA256,
        vision_dir / "config.json": DINO_CONFIG_SHA256,
        audio_dir / "audio.pt": CLAP_WEIGHT_SHA256,
        audio_dir / "config.json": CLAP_CONFIG_SHA256,
        clap_root / "model" / "preprocessor_config.json": CLAP_FRONTEND_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"pinned frozen-encoder artifact changed: {path}")

    image_processor = AutoImageProcessor.from_pretrained(
        str(vision_dir), local_files_only=True, trust_remote_code=False
    )
    vision = AutoModel.from_pretrained(
        str(vision_dir), local_files_only=True, trust_remote_code=False
    ).to(device)
    vision.eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in vision.parameters()) != 22_056_576:
        raise ValueError("DINOv2 parameter inventory changed")

    audio = ClapAudioModelWithProjection(
        ClapAudioConfig.from_pretrained(audio_dir, local_files_only=True)
    )
    audio.load_state_dict(
        torch.load(audio_dir / "audio.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    audio.to(device).eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in audio.parameters()) != 28_190_872:
        raise ValueError("CLAP audio parameter inventory changed")
    audio_frontend = ClapFeatureExtractor.from_pretrained(
        str(clap_root / "model"), local_files_only=True
    )
    if audio_frontend.sampling_rate != TARGET_RATE or audio_frontend.nb_max_samples != MAX_SAMPLES:
        raise ValueError("CLAP frontend window contract changed")
    return image_processor, vision, audio_frontend, audio


def _decode_selected_rgb(media_path: Path, frame_ids: set[int]):
    import av

    selected = {}
    maximum = max(frame_ids)
    with av.open(str(media_path)) as container:
        if not container.streams.video:
            raise ValueError("acquired source clip has no video stream")
        for index, frame in enumerate(container.decode(video=0)):
            if index in frame_ids:
                selected[index] = frame.to_ndarray(format="rgb24")
            if index >= maximum:
                break
    if selected.keys() != frame_ids:
        raise ValueError("one or more selected annotation frames are absent from decoded video")
    return selected


def _decode_resampled_audio(media_path: Path):
    import av

    with av.open(str(media_path)) as container:
        if not container.streams.audio:
            raise ValueError("acquired source clip has no audio stream")
        native_rate = container.streams.audio[0].codec_context.sample_rate
        if native_rate <= 0:
            raise ValueError("invalid native audio sample rate")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_RATE)
        arrays = []
        start = None
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                if start is None:
                    if converted.pts is None or converted.time_base is None:
                        raise ValueError("resampled audio lacks actual PTS")
                    start = float(converted.pts * converted.time_base)
                arrays.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            arrays.append(converted.to_ndarray())
    if not arrays or start is None:
        raise ValueError("resampled audio stream is empty")
    waveform = np.concatenate(arrays, axis=1)
    if waveform.shape[0] != 1 or not np.isfinite(waveform).all():
        raise ValueError("resampled audio must be finite mono samples")
    return waveform[0], start


def _encode_visual_event(crops, image_processor, vision, device: str, batch_size: int):
    import torch

    features = []
    for offset in range(0, len(crops), batch_size):
        inputs = image_processor(images=crops[offset : offset + batch_size], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            encoded = vision(**inputs).last_hidden_state
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("DINOv2 emitted non-finite crop features")
        features.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(features, axis=0)


def _encode_audio_event(waveform, start_time, event_row, audio_frontend, audio_model, device: str):
    import torch
    from torch.nn import functional as F

    event_start = event_row.get("event_start_seconds")
    event_end = event_row.get("event_end_seconds")
    if event_row.get("status") == "BOUNDARY_REVIEW":
        event_start, event_end = event_row["clipped_window_seconds"]
    left, right = event_sample_bounds(
        event_start, event_end, start_time, TARGET_RATE, len(waveform)
    )
    event_wave = waveform[left:right]
    spans = window_spans(len(event_wave), MAX_SAMPLES)
    vectors = []
    for begin, stop in spans:
        prepared = audio_frontend(
            event_wave[begin:stop], sampling_rate=TARGET_RATE, return_tensors="pt"
        )
        if prepared["is_longer"].any():
            raise ValueError("CLAP frontend would truncate a deterministic event window")
        prepared = {key: value.to(device) for key, value in prepared.items()}
        with torch.inference_mode():
            vector = F.normalize(audio_model(**prepared).audio_embeds.float(), dim=-1)
        if not torch.isfinite(vector).all():
            raise FloatingPointError("CLAP emitted non-finite audio features")
        vectors.append(vector.cpu().numpy()[0])
    pooled = duration_weighted_feature(
        np.stack(vectors), [stop - begin for begin, stop in spans]
    )
    return pooled, left, right


def extract_features(
    *,
    plan_path: Path,
    train_source: Path,
    valid_source: Path,
    train_manifest: Path,
    valid_manifest: Path,
    pts_receipt: Path,
    vision_dir: Path,
    clap_root: Path,
    output_dir: Path,
    device: str = "cuda",
    image_batch_size: int = 16,
):
    import torch
    from PIL import Image

    if output_dir.exists():
        raise ValueError("fresh output directory required")
    if image_batch_size < 1:
        raise ValueError("image batch size must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != PLAN_SHA256:
        raise ValueError("media plan hash differs from pinned plan")
    train_video_ids = {
        row["video_id"]
        for row in plan["splits"]["train"]["rows"]
        if row["model_split"] == "train"
    }
    if len(train_video_ids) != 300:
        raise ValueError("pinned plan must contain exactly 300 model-training videos")
    selection_video_ids = model_selection_video_ids(train_video_ids, selection_count=60)
    pts = json.loads(pts_receipt.read_text(encoding="utf-8"))
    if pts.get("plan_sha256") != PLAN_SHA256 or len(pts.get("reports", [])) != 2:
        raise ValueError("complete PTS diagnostic receipt is not pinned to this plan")
    audit_by_event = {}
    for report in pts["reports"]:
        for event in report["event_rows"]:
            key = (report["split"], event["video_id"], event["sound_id"])
            if key in audit_by_event:
                raise ValueError("duplicate event in PTS diagnostic receipt")
            audit_by_event[key] = event

    image_processor, vision, audio_frontend, audio_model = _load_feature_models(
        vision_dir, clap_root, device
    )

    sources = {
        "train": (train_source, train_manifest, TRAIN_SOURCE_SHA256),
        "valid": (valid_source, valid_manifest, VALID_SOURCE_SHA256),
    }
    all_rows = []
    input_hashes = {"plan": sha256_file(plan_path), "pts_diagnostic": sha256_file(pts_receipt)}
    for split, (source_root, media_manifest_path, pins) in sources.items():
        manifest_bytes = media_manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest.get("split") != split or manifest.get("plan_sha256") != PLAN_SHA256:
            raise ValueError("media manifest split/plan pin mismatch")
        if manifest.get("status") != "PASS_ACQUISITION_NOT_DECODE":
            raise ValueError("feature extraction requires terminal media acquisition")
        expected_manifest_sha = next(
            report["media_manifest_sha256"]
            for report in pts["reports"]
            if report["split"] == split
        )
        if manifest_sha != expected_manifest_sha:
            raise ValueError("media manifest differs from full PTS diagnostic snapshot")
        data, cuts, source_hashes = load_annotations(
            source_root / f"{split}_annotations.zip",
            source_root / f"cut_frame_mapping_{split}.json",
            split,
            {"annotations": pins["annotations"], "cuts": pins["cuts"]},
        )
        input_hashes[f"{split}_annotations"] = source_hashes["annotations"]
        input_hashes[f"{split}_cuts"] = source_hashes["cuts"]
        plan_rows = {row["video_id"]: row for row in plan["splits"][split]["rows"]}
        clips = {row["video_id"]: row for row in manifest["clips"]}
        events_by_video = defaultdict(list)
        for event in summarize_split(data, cuts, split)["candidate_records"]:
            if event["video_id"] in plan_rows:
                events_by_video[event["video_id"]].append(event)
        if set(events_by_video) != set(clips):
            raise ValueError("media and eligible-event video inventories differ")
        for video_index, video_id in enumerate(sorted(events_by_video), 1):
            clip = clips[video_id]
            media_path = Path(clip["path"])
            if not media_path.is_file() or sha256_file(media_path) != clip["sha256"]:
                raise ValueError(f"media file missing or hash-mismatched: {video_id}")
            events = sorted(events_by_video[video_id], key=lambda row: row["sound_id"])
            frame_ids = {event["selected_frame"] for event in events}
            rgb_by_frame = _decode_selected_rgb(media_path, frame_ids)
            timing_rows = [
                audit_by_event.get((split, video_id, event["sound_id"])) for event in events
            ]
            if any(row is None for row in timing_rows):
                raise ValueError(f"PTS receipt is missing events for {video_id}")
            waveform, resampled_start = _decode_resampled_audio(media_path)
            tracks = {track["id"]: track for track in data[video_id]["object_tracking"]}
            for event, audit_row in zip(events, timing_rows, strict=True):
                ordered_ids = deterministic_candidate_order(
                    video_id, event["sound_id"], event["candidate_track_ids"]
                )
                source_ids = event["source_parent_ids"]
                if len(source_ids) != 1 or source_ids[0] not in ordered_ids:
                    raise ValueError("event must have exactly one source candidate")
                target_index = target_index_for_candidate_order(ordered_ids, source_ids[0])
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
                event_key = f"{split}-{video_id}-sound-{event['sound_id']}"
                audio_path = output_dir / f"{event_key}-audio.npy"
                visual_path = output_dir / f"{event_key}-visual.npy"
                atomic_save_npy(audio_path, audio)
                atomic_save_npy(visual_path, visual)
                candidate_digest = hashlib.sha256(
                    ",".join(str(value) for value in ordered_ids).encode()
                ).hexdigest()
                all_rows.append(
                    {
                        "id": f"{split}:{video_id}:{event['sound_id']}",
                        "video_id": video_id,
                        "sound_id": event["sound_id"],
                        "split": (
                            "selection"
                            if video_id in selection_video_ids
                            else "fit"
                            if plan_rows[video_id]["model_split"] == "train"
                            else plan_rows[video_id]["model_split"]
                        ),
                        "target_index": target_index,
                        "candidate_count": len(ordered_ids),
                        "candidate_order_sha256": candidate_digest,
                        "candidate_order_ids_for_audit_only": ordered_ids,
                        "media_sha256": clip["sha256"],
                        "annotation_sha256": source_hashes["annotations"],
                        "right_censored_at_media_end": bool(
                            audit_row.get("status") == "BOUNDARY_REVIEW"
                        ),
                        "audio_sample_start": sample_start,
                        "audio_sample_end_exclusive": sample_end,
                        "audio_feature_path": str(audio_path.resolve()),
                        "audio_feature_sha256": sha256_file(audio_path),
                        "visual_feature_path": str(visual_path.resolve()),
                        "visual_feature_sha256": sha256_file(visual_path),
                    }
                )
            atomic_write_json(
                output_dir / "progress.json",
                {
                    "status": "RUNNING",
                    "source_split": split,
                    "completed_videos_in_current_split": video_index,
                    "total_videos_in_current_split": len(events_by_video),
                    "feature_rows_so_far": len(all_rows),
                },
            )

    ids = [row["id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("feature event IDs are not unique")
    allowed_splits = {"fit", "selection", "calibration", "development_validation"}
    if any(row["split"] not in allowed_splits for row in all_rows):
        raise ValueError("feature rows contain an unrecognized model split")
    manifest_path = output_dir / "features.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "status": "PASS_FROZEN_FEATURE_EXTRACTION",
        "rows": len(all_rows),
        "rows_by_split": {
            name: sum(row["split"] == name for row in all_rows)
            for name in ("fit", "selection", "calibration", "development_validation")
        },
        "right_censored_rows": sum(row["right_censored_at_media_end"] for row in all_rows),
        "input_hashes": input_hashes,
        "feature_manifest": str(manifest_path.resolve()),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "vision_model_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "vision_weights_sha256": VISION_WEIGHT_SHA256,
        "audio_model_revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        "audio_weights_sha256": CLAP_WEIGHT_SHA256,
        "vision_parameters_frozen": sum(parameter.numel() for parameter in vision.parameters()),
        "audio_parameters_frozen": sum(parameter.numel() for parameter in audio_model.parameters()),
        "device": device,
        "dtype": "float32",
        "image_batch_size": image_batch_size,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pyav_version": __import__("av").__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else 0
        ),
        "gpu_used": device == "cuda",
        "training": False,
        "test_opened": False,
    }
    atomic_write_json(output_dir / "result.json", summary)
    atomic_write_json(
        output_dir / "progress.json",
        {"status": "COMPLETE", "feature_rows": len(all_rows)},
    )
    return summary


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--pts-receipt", type=Path, required=True)
    parser.add_argument("--vision-dir", type=Path, required=True)
    parser.add_argument("--clap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=16)
    args = parser.parse_args()
    result = extract_features(
        plan_path=args.plan,
        train_source=args.train_source,
        valid_source=args.valid_source,
        train_manifest=args.train_manifest,
        valid_manifest=args.valid_manifest,
        pts_receipt=args.pts_receipt,
        vision_dir=args.vision_dir,
        clap_root=args.clap_root,
        output_dir=args.output_dir,
        device=args.device,
        image_batch_size=args.image_batch_size,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    _main()
