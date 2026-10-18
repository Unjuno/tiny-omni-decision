"""CPU smoke for annotation-box crops through the pinned DINOv2-small encoder."""

import argparse
import json
import platform
import time
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import (
    TRAIN_SOURCE_SHA256,
    load_annotations,
    summarize_split,
)
from scripts.perception_av_event_protocol import deterministic_candidate_order

PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"
VISION_WEIGHT_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
PROCESSOR_SHA256 = "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"


def normalized_box_to_pixels(box, height, width):
    import math

    if (
        len(box) != 4
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box)
    ):
        raise ValueError("box must contain four finite normalized coordinates")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("normalized box must be ordered and within [0, 1]")
    left = math.floor(x0 * width)
    top = math.floor(y0 * height)
    right = math.ceil(x1 * width)
    bottom = math.ceil(y1 * height)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("pixel crop is empty or outside the decoded image")
    return left, top, right, bottom


def track_box_at_frame(track, frame_id):
    frame_ids = track.get("frame_ids")
    if not isinstance(frame_ids, list) or frame_ids.count(frame_id) != 1:
        raise ValueError("candidate track must have exactly one annotation at selected frame")
    index = frame_ids.index(frame_id)
    boxes = track.get("bounding_boxes")
    masked = track.get("is_masked")
    if (
        not isinstance(boxes, list)
        or not isinstance(masked, list)
        or len(boxes) != len(frame_ids)
        or len(masked) != len(frame_ids)
        or masked[index] != 0
    ):
        raise ValueError("candidate crop must be an unmasked aligned annotation")
    return boxes[index]


def run_smoke(train_source, media_manifest_path, model_dir, output_dir):
    import av
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    train_source = Path(train_source)
    media_manifest_path = Path(media_manifest_path)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("fresh output directory required")
    media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    if media_manifest.get("split") != "train" or media_manifest.get("plan_sha256") != PLAN_SHA256:
        raise ValueError("expected pinned TRAIN media manifest")
    clips = {clip["video_id"]: clip for clip in media_manifest.get("clips", [])}
    if "video_11362" not in clips:
        raise ValueError("fixed training smoke video is not yet acquired")
    clip = clips["video_11362"]
    media_path = Path(clip["path"])
    if sha256_file(media_path) != clip.get("sha256"):
        raise ValueError("smoke media SHA-256 mismatch")
    if sha256_file(model_dir / "model.safetensors") != VISION_WEIGHT_SHA256:
        raise ValueError("pinned DINOv2 weights changed")
    if sha256_file(model_dir / "preprocessor_config.json") != PROCESSOR_SHA256:
        raise ValueError("pinned DINOv2 processor changed")
    if sha256_file(model_dir / "config.json") != CONFIG_SHA256:
        raise ValueError("pinned DINOv2 config changed")

    annotation_path = train_source / "train_annotations.zip"
    cut_path = train_source / "cut_frame_mapping_train.json"
    data, cuts, source_pins = load_annotations(
        annotation_path,
        cut_path,
        "train",
        {
            "annotations": TRAIN_SOURCE_SHA256["annotations"],
            "cuts": TRAIN_SOURCE_SHA256["cuts"],
        },
    )
    candidates = summarize_split(data, cuts, "train")["candidate_records"]
    event = next(
        row
        for row in candidates
        if row["video_id"] == "video_11362" and row["sound_id"] == 4
    )
    metadata = data[event["video_id"]]["metadata"]
    tracks_by_id = {
        track["id"]: track for track in data[event["video_id"]]["object_tracking"]
    }
    ordered_track_ids = deterministic_candidate_order(
        event["video_id"], event["sound_id"], event["candidate_track_ids"]
    )
    started = time.perf_counter()
    with av.open(str(media_path)) as container:
        video_frames = list(container.decode(video=0))
    if len(video_frames) != metadata["num_frames"]:
        raise ValueError("decoded frame count differs from annotation metadata")
    frame = video_frames[event["selected_frame"]]
    array = frame.to_ndarray(format="rgb24")
    height, width = array.shape[:2]
    expected_height, expected_width = metadata["resolution"]
    if (height, width) != (expected_height, expected_width):
        raise ValueError("decoded RGB shape differs from annotation resolution")
    image = Image.fromarray(array)

    crops = []
    crop_boxes = []
    for track_id in ordered_track_ids:
        box = track_box_at_frame(tracks_by_id[track_id], event["selected_frame"])
        pixel_box = normalized_box_to_pixels(box, height, width)
        crop = image.crop(pixel_box)
        if crop.width <= 0 or crop.height <= 0:
            raise ValueError("empty candidate image crop")
        crops.append(crop)
        crop_boxes.append(pixel_box)

    torch.set_num_threads(2)
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    ).eval()
    inputs = processor(images=crops, return_tensors="pt")
    with torch.inference_mode():
        first = model(**inputs).last_hidden_state
        second = model(**inputs).last_hidden_state
    if not torch.isfinite(first).all() or not torch.equal(first, second):
        raise ValueError("frozen crop embeddings are non-finite or nondeterministic")
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / "candidate-features.npy"
    np.save(feature_path, first.cpu().numpy())
    output = {
        "status": "PASS_TRAIN_CROP_ENCODER_SMOKE",
        "task": "perception-av-event-source-selection-v1",
        "scope": "one TRAIN event; no target/source labels passed to image encoder",
        "source_annotation_sha256": source_pins,
        "media_video_id": event["video_id"],
        "media_sha256": clip["sha256"],
        "sound_id_for_join_only": event["sound_id"],
        "candidate_count": len(crops),
        "selected_frame_id": event["selected_frame"],
        "decoded_video_frames": len(video_frames),
        "decoded_frame_shape": list(array.shape),
        "candidate_crop_pixel_boxes_in_deterministic_order": crop_boxes,
        "model_repo": "facebook/dinov2-small",
        "model_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_shape": list(first.shape),
        "finite": bool(torch.isfinite(first).all()),
        "repeat_exact": bool(torch.equal(first, second)),
        "feature_file": feature_path.name,
        "feature_file_sha256": sha256_file(feature_path),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pyav_version": av.__version__,
        "python_version": platform.python_version(),
        "device": "cpu",
        "dtype": str(first.dtype),
        "seconds": time.perf_counter() - started,
        "training": False,
        "model_selection": False,
        "quality_evaluated": False,
        "gpu_used": False,
        "test_opened": False,
    }
    output["script_sha256"] = sha256_file(Path(__file__))
    output_dir.joinpath("result.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(args.train_source, args.media_manifest, args.model_dir, args.output_dir),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
