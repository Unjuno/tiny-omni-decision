"""Label-blind, video-only DINO similarity audit for a frozen PhysicalAI cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

FRACTIONS = (0.2, 0.5, 0.8)
MODEL_REVISION = "facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MANIFEST_SHA256 = "fb51eb0028c8f55372930548d2e480bbf616debc1a59779f29098d86ce1bbe7f"
RECEIPT_SHA256 = "b7a40364f1f931e897430e3057379cbd52ebaa4e1aae823c266d33dcb7e4f634"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_frame_indices(frame_count: int) -> tuple[int, int, int]:
    if frame_count < 3:
        raise ValueError("video must expose at least three frames")
    indices = tuple(round((frame_count - 1) * fraction) for fraction in FRACTIONS)
    if len(set(indices)) != 3:
        raise ValueError("video is too short for three distinct temporal samples")
    return indices


def render_neighbor_contact_sheet(
    thumbnails: np.ndarray,
    episode_ids: np.ndarray,
    splits: np.ndarray,
    pairs: list[dict[str, Any]],
):
    from PIL import Image, ImageDraw

    lookup = {int(episode): index for index, episode in enumerate(episode_ids)}
    selected = pairs[:12]
    cell = 72
    label_height = 20
    pair_height = label_height + cell
    sheet = Image.new("RGB", (6 * cell, len(selected) * pair_height), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(selected):
        y = row * pair_height
        draw.text(
            (2, y + 2),
            "val {} ↔ train {} | cosine {:.4f}".format(
                pair["validation_episode"],
                pair["train_episode"],
                pair["cosine_similarity"],
            ),
            fill="black",
        )
        for column, episode in enumerate(
            (pair["validation_episode"], pair["train_episode"])
        ):
            index = lookup[int(episode)]
            expected_split = "validation" if column == 0 else "train"
            if str(splits[index]) != expected_split:
                raise ValueError("neighbor contact-sheet split mismatch")
            for frame_index in range(3):
                image = Image.fromarray(thumbnails[index, frame_index]).resize(
                    (cell, cell), Image.Resampling.NEAREST
                )
                sheet.paste(image, (column * 3 * cell + frame_index * cell, y + label_height))
    return sheet


def representative_sample_ids(episode_ids: np.ndarray, splits: np.ndarray, count: int = 30):
    selected: dict[str, list[int]] = {}
    for split in ("train", "validation"):
        candidates = [
            int(episode)
            for episode, value in zip(episode_ids, splits, strict=True)
            if str(value) == split
        ]
        candidates.sort(
            key=lambda episode: hashlib.sha256(
                f"physicalai-scene-audit-seed17|{split}|{episode}".encode()
            ).hexdigest()
        )
        if len(candidates) < count:
            raise ValueError(f"not enough {split} episodes for representative sample")
        selected[split] = candidates[:count]
    return selected


def render_representative_contact_sheet(
    thumbnails: np.ndarray,
    episode_ids: np.ndarray,
    splits: np.ndarray,
    selected: dict[str, list[int]],
):
    from PIL import Image, ImageDraw

    lookup = {int(episode): index for index, episode in enumerate(episode_ids)}
    sample_ids = [
        (split, episode)
        for split in ("train", "validation")
        for episode in selected[split]
    ]
    cell = 64
    row_height = 82
    sheet = Image.new("RGB", (3 * cell, len(sample_ids) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (split, episode) in enumerate(sample_ids):
        index = lookup[episode]
        if str(splits[index]) != split:
            raise ValueError("representative contact-sheet split mismatch")
        y = row * row_height
        draw.text((2, y + 1), f"{split} episode {episode}", fill="black")
        for frame_index in range(3):
            image = Image.fromarray(thumbnails[index, frame_index]).resize(
                (cell, cell), Image.Resampling.NEAREST
            )
            sheet.paste(image, (frame_index * cell, y + 18))
    return sheet


def validate_inputs(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    *,
    receipt_sha: str,
    manifest_sha: str,
    expected_receipt_sha: str | None = None,
    expected_manifest_sha: str | None = None,
) -> list[dict[str, Any]]:
    if receipt.get("dataset_id") != "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm":
        raise ValueError("unexpected dataset")
    if receipt.get("revision") != "73c00c18e7ca108e91846294b20a1d7cd5de9340":
        raise ValueError("unexpected pinned source revision")
    if receipt.get("subset") != "panda-stack-platforms":
        raise ValueError("unexpected source subset")
    if expected_receipt_sha is not None and receipt_sha != expected_receipt_sha:
        raise ValueError("receipt hash differs from the pinned acquisition receipt")
    if expected_manifest_sha is not None and manifest_sha != expected_manifest_sha:
        raise ValueError("manifest hash differs from the frozen split manifest")
    if receipt.get("split_manifest_sha256") != manifest_sha:
        raise ValueError("receipt does not bind to the supplied split manifest")
    if manifest.get("status") != "REVIEW_NOT_ADMITTED":
        raise ValueError("candidate status changed; require a new review")
    selection = receipt.get("selection", {})
    train = selection.get("train_episode_indices", [])
    validation = selection.get("validation_episode_indices", [])
    reserved = set(manifest.get("reserved_development_episode_indices", []))
    if len(train) != 4000 or len(validation) != 1000:
        raise ValueError("the frozen 4000/1000 train/validation cohort is required")
    if set(train) & set(validation) or (set(train) | set(validation)) & reserved:
        raise ValueError("split or reserved-holdout overlap")
    assignments = {
        int(row["episode_index"]): row for row in manifest.get("episode_assignments", [])
    }
    files = receipt.get("files", [])
    videos = [row for row in files if row.get("kind") == "video"]
    if len(videos) != 5000 or len(files) != 10000:
        raise ValueError("receipt must contain exactly 5000 videos and paired parquet records")
    if receipt.get("reserved_holdout_payload_downloaded") is not False:
        raise ValueError("reserved cohort must remain unopened")
    expected = {episode: "train" for episode in train}
    expected.update({episode: "validation" for episode in validation})
    seen: set[int] = set()
    ordered: list[dict[str, Any]] = []
    for row in videos:
        episode = int(row["episode"])
        source = assignments.get(episode)
        if episode in seen or expected.get(episode) != row.get("split") or source is None:
            raise ValueError("video receipt split/identity mismatch")
        if (
            source.get("split") != row.get("split")
            or source.get("video_sha256") != row.get("sha256")
            or source.get("video_bytes") != row.get("size")
            or source.get("video_path") != row.get("path")
        ):
            raise ValueError("video receipt does not match pinned split manifest")
        seen.add(episode)
        ordered.append(row)
    if seen != set(expected):
        raise ValueError("video receipt is missing selected train/validation identities")
    return sorted(ordered, key=lambda row: (row["split"], int(row["episode"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("optional split limit must be positive")

    import av
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    receipt_bytes = args.receipt.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    receipt = json.loads(receipt_bytes)
    manifest = json.loads(manifest_bytes)
    rows = validate_inputs(
        receipt,
        manifest,
        receipt_sha=hashlib.sha256(receipt_bytes).hexdigest(),
        manifest_sha=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_receipt_sha=RECEIPT_SHA256,
        expected_manifest_sha=MANIFEST_SHA256,
    )
    if args.limit_per_split is not None:
        rows = [
            row
            for split in ("train", "validation")
            for row in [x for x in rows if x["split"] == split][: args.limit_per_split]
        ]
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if not args.model_dir.is_dir():
        parser.error("pinned local DINOv2 model directory is required; network loading is disabled")
    model_files = {}
    for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
        path = args.model_dir / name
        if not path.is_file():
            parser.error(f"missing local model file: {name}")
        model_files[name] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
    if model_files["model.safetensors"]["sha256"] != (
        "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
    ):
        parser.error("local DINOv2 checkpoint hash differs from the pinned audited weight")

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).eval()
    device = torch.device(args.device)
    model.to(device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ids: list[int] = []
    splits: list[str] = []
    vectors: list[np.ndarray] = []
    thumbnails: list[np.ndarray] = []
    decode_seconds = 0.0
    inference_seconds = 0.0
    for ordinal, record in enumerate(rows, start=1):
        path = Path(record["local_path"])
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"video missing or size mismatch: episode {record['episode']}")
        if digest_file(path) != record["sha256"]:
            raise ValueError(f"video publisher hash mismatch: episode {record['episode']}")
        decode_started = time.perf_counter()
        frames: list[Image.Image] = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frame_count = int(stream.frames)
            indices = set(selected_frame_indices(frame_count))
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in indices:
                    frames.append(frame.to_image().convert("RGB"))
                    if len(frames) == 3:
                        break
        decode_seconds += time.perf_counter() - decode_started
        if len(frames) != 3:
            raise ValueError(f"could not decode three fixed temporal samples: {record['episode']}")
        thumb = [np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR)) for image in frames]
        thumbnails.append(np.stack(thumb).astype(np.uint8))
        infer_started = time.perf_counter()
        inputs = processor(images=frames, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
        inference_seconds += time.perf_counter() - infer_started
        vectors.append(output)
        ids.append(int(record["episode"]))
        splits.append(str(record["split"]))
        if ordinal % 100 == 0 or ordinal == len(rows):
            progress = {
                "processed_videos": ordinal,
                "total_videos": len(rows),
                "elapsed_seconds": time.perf_counter() - started,
                "decode_seconds": decode_seconds,
                "inference_seconds": inference_seconds,
            }
            (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)

    feature_array = np.stack(vectors).astype(np.float32)
    thumbnail_array = np.stack(thumbnails)
    id_array = np.asarray(ids, dtype=np.int32)
    split_array = np.asarray(splits)
    train_mask = split_array == "train"
    validation_mask = split_array == "validation"
    pooled = feature_array.mean(axis=1)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    train_vectors = pooled[train_mask]
    validation_vectors = pooled[validation_mask]
    similarities = validation_vectors @ train_vectors.T
    flat_order = np.argsort(similarities.ravel())[::-1][:100]
    pair_indices = np.unravel_index(flat_order, similarities.shape)
    pairs = [
        {
            "validation_episode": int(id_array[validation_mask][v]),
            "train_episode": int(id_array[train_mask][t]),
            "cosine_similarity": float(similarities[v, t]),
        }
        for v, t in zip(pair_indices[0], pair_indices[1], strict=True)
    ]
    contact_sheet = render_neighbor_contact_sheet(
        thumbnail_array, id_array, split_array, pairs
    )
    contact_sheet.save(args.output / "top-neighbor-contact-sheet.png")
    representative_count = min(30, int(train_mask.sum()), int(validation_mask.sum()))
    representative_ids = representative_sample_ids(
        id_array, split_array, count=representative_count
    )
    representative_sheet = render_representative_contact_sheet(
        thumbnail_array, id_array, split_array, representative_ids
    )
    representative_sheet.save(args.output / "representative-contact-sheet.png")
    values = similarities.ravel()
    report = {
        "status": "PASS_LABEL_BLIND_VIDEO_SIMILARITY_AUDIT",
        "source_revision": receipt["revision"],
        "subset": receipt["subset"],
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "model_revision": MODEL_REVISION,
        "model_files": model_files,
        "processor_sha256": digest_file(args.model_dir / "preprocessor_config.json"),
        "frames_per_video": 3,
        "frames_per_forward_call": 3,
        "frame_sampling": "decoded frame indices nearest 20%, 50%, 80% of declared frame count",
        "episodes": len(ids),
        "train_episodes": int(train_mask.sum()),
        "validation_episodes": int(validation_mask.sum()),
        "limit_per_split": args.limit_per_split,
        "feature_shape": list(feature_array.shape),
        "cross_split_pair_count": int(similarities.size),
        "cross_split_cosine": {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
            "pairs_at_or_above_threshold": {
                str(threshold): int(np.sum(values >= threshold))
                for threshold in (0.98, 0.99, 0.995)
            },
        },
        "top_100_cross_split_neighbors": pairs,
        "representative_video_sample": {
            "method": "SHA-256 order of seed17|split|episode; first 30 per split",
            "train_episode_ids": representative_ids["train"],
            "validation_episode_ids": representative_ids["validation"],
            "contact_sheet_sha256": digest_file(
                args.output / "representative-contact-sheet.png"
            ),
        },
        "decoded_source_files": len(ids),
        "reserved_payload_opened": False,
        "labels_or_parquet_opened": False,
        "features_fit_or_quality_evaluated": False,
        "device": str(device),
        "dtype": "float32",
        "decode_seconds": decode_seconds,
        "inference_seconds": inference_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "interpretation_limit": (
            "DINO similarity is exploratory, not a scene identity or validated grouping rule. "
            "A human-blind review of frozen contact sheets is required before any split change."
        ),
    }
    np.savez_compressed(
        args.output / "scene-similarity-features.npz",
        episode=id_array,
        split=split_array,
        frame_features=feature_array,
        thumbnails=thumbnail_array,
    )
    (args.output / "scene-similarity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary_keys = ("status", "episodes", "cross_split_cosine", "total_wall_seconds")
    print(json.dumps({key: report[key] for key in summary_keys}))


if __name__ == "__main__":
    main()
