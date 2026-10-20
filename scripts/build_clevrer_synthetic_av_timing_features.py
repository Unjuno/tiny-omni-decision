"""Build frozen MiniLM text and waveform-derived audio features for CLEVRER AV timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.create_clevrer_synthetic_av_protocol import DATASET, REVISION

VIDEO_FRAME_COUNT = 128
VIDEO_FRAME_RATE = 25.6
AUDIO_SAMPLE_RATE = 16_000
SAMPLES_PER_VIDEO_FRAME = 625
TONE_FREQUENCY_HZ = 880
TONE_DURATION_MS = 80
TEXT_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TEXT_MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
TEXT_PARAMETER_COUNT = 17_389_824


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def audio_energy_features(sound_onset_sample: int, video_start_frame: int) -> np.ndarray:
    """Render the frozen tone burst and pool its waveform energy into 16 aligned bins."""
    if (
        isinstance(sound_onset_sample, bool)
        or not isinstance(sound_onset_sample, int)
        or isinstance(video_start_frame, bool)
        or not isinstance(video_start_frame, int)
    ):
        raise ValueError("audio onset and video start must be integer frame/sample indices")
    onset = sound_onset_sample - video_start_frame * SAMPLES_PER_VIDEO_FRAME
    clip_samples = VIDEO_FRAME_COUNT * SAMPLES_PER_VIDEO_FRAME
    tone_samples = round(AUDIO_SAMPLE_RATE * TONE_DURATION_MS / 1000)
    if onset < 0 or onset >= clip_samples:
        raise ValueError("audio onset must fall inside the aligned 128-frame video clip")

    # Keep the full 80 ms tail even when a valid onset is near the final video frame.
    waveform = np.zeros(clip_samples + tone_samples, dtype=np.float32)
    time = np.arange(tone_samples, dtype=np.float32) / AUDIO_SAMPLE_RATE
    envelope = np.hanning(tone_samples).astype(np.float32)
    burst = 0.3 * np.sin(2 * np.pi * TONE_FREQUENCY_HZ * time) * envelope
    waveform[onset : onset + tone_samples] = burst

    # Each token covers 8 source video frames; the last bin also includes the
    # physical audio tail after frame 127. Total energy is normalized per clip.
    edges = [index * 8 * SAMPLES_PER_VIDEO_FRAME for index in range(17)]
    edges[-1] = len(waveform)
    energy = np.asarray(
        [
            np.square(waveform[left:right], dtype=np.float64).sum()
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ],
        dtype=np.float64,
    )
    total_energy = float(energy.sum())
    if total_energy <= 0 or not np.isfinite(energy).all():
        raise ValueError("rendered audio has no finite tone energy")
    result = (energy / total_energy).astype(np.float32).reshape(16, 1)
    if not np.isfinite(result).all() or not np.isclose(result.sum(), 1.0, atol=1e-6):
        raise ValueError("invalid normalized 16-bin audio energy features")
    return result


def _read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError(f"{path} must contain nonempty {split} rows")
    return rows


def _validate_rows(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    scene_hashes: dict[str, str] = {}
    pairs: dict[str, list[dict[str, Any]]] = {}
    for split, rows in (("train", train), ("validation", validation)):
        for row in rows:
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError("sample IDs must be nonempty and globally unique")
            ids.add(sample_id)
            if row.get("source") != f"{DATASET}@{REVISION}":
                raise ValueError(f"unexpected source revision in {sample_id}")
            scene = str(row.get("scene_group_id", ""))
            media = str(row.get("media_sha256", ""))
            if not scene or not media:
                raise ValueError(f"missing scene/media identity in {sample_id}")
            if scene in scene_hashes and scene_hashes[scene] != media:
                raise ValueError(f"scene maps to multiple media assets: {scene}")
            scene_hashes[scene] = media
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError(f"missing pair ID in {sample_id}")
            pairs.setdefault(pair_id, []).append(row)
            options = row.get("options")
            target = row.get("target_index")
            frame_range = row.get("video_frame_range")
            if (
                row.get("split") != split
                or not isinstance(options, list)
                or options != ["sound before collision", "sound after collision"]
                or target not in (0, 1)
                or row.get("target") != options[target]
                or not isinstance(frame_range, list)
                or len(frame_range) != 2
                or frame_range[1] - frame_range[0] != VIDEO_FRAME_COUNT - 1
            ):
                raise ValueError(f"invalid frozen binary timing row: {sample_id}")
    for pair_id, members in pairs.items():
        if (
            len(members) != 2
            or {row["target_index"] for row in members} != {0, 1}
            or len({row["question"] for row in members}) != 1
            or len({row["scene_group_id"] for row in members}) != 1
            or len({row["media_sha256"] for row in members}) != 1
            or len({row["collision_frame"] for row in members}) != 1
            or len({row["video_frame_range"][0] for row in members}) != 1
            or len({row["sound_onset_sample"] for row in members}) != 2
        ):
            raise ValueError(f"paired timing variants are incomplete or inconsistent: {pair_id}")
    train_scenes = {row["scene_group_id"] for row in train}
    validation_scenes = {row["scene_group_id"] for row in validation}
    train_media = {row["media_sha256"] for row in train}
    validation_media = {row["media_sha256"] for row in validation}
    if train_scenes & validation_scenes or train_media & validation_media:
        raise ValueError("train and validation scenes/media must be disjoint")


def build_features(
    *, sample_dir: Path, video_cache_dir: Path, text_model_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output_dir}")
    train = _read_rows(sample_dir / "train.jsonl", "train")
    validation = _read_rows(sample_dir / "validation.jsonl", "validation")
    _validate_rows(train, validation)

    cache_report_path = video_cache_dir / "cache-report.json"
    video_cache_path = video_cache_dir / "framewise-features.npz"
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if (
        cache_report.get("status") != "complete_videoprism_framewise_feature_cache"
        or cache_report.get("model_revision") != "be719a406d563b66f0ac969e7c94bab8e997c81a"
        or cache_report.get("checkpoint_sha256")
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        or cache_report.get("frame_count") != 16
        or sha256_file(video_cache_path) != cache_report.get("feature_cache_sha256")
    ):
        raise ValueError("VideoPrism feature cache does not match the frozen 16-frame source")
    with np.load(video_cache_path, allow_pickle=False) as archive:
        media_hashes = archive["media_sha256"].astype(str)
        video_values = archive["features"].astype(np.float32, copy=False)
    if (
        video_values.shape != (len(media_hashes), 16, 768)
        or len(set(media_hashes.tolist())) != len(media_hashes)
        or not np.isfinite(video_values).all()
    ):
        raise ValueError("VideoPrism feature cache has invalid shape, identity or values")
    video_by_media = dict(zip(media_hashes.tolist(), video_values, strict=True))
    rows = train + validation
    if any(row["media_sha256"] not in video_by_media for row in rows):
        raise ValueError("VideoPrism cache does not cover every frozen train/validation sample")

    model_path = text_model_dir / "model.safetensors"
    if sha256_file(model_path) != TEXT_MODEL_SHA256:
        raise ValueError("MiniLM checkpoint SHA-256 differs from the pinned English encoder")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for the pinned MiniLM text features"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
    text_model = AutoModel.from_pretrained(text_model_dir, local_files_only=True).cpu().eval()
    parameter_count = sum(parameter.numel() for parameter in text_model.parameters())
    if parameter_count != TEXT_PARAMETER_COUNT:
        raise ValueError(f"unexpected MiniLM parameter count: {parameter_count}")

    unique_questions = sorted({str(row["question"]) for row in rows})
    import torch

    question_vectors: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(unique_questions), 64):
            texts = unique_questions[start : start + 64]
            tokens = tokenizer(
                texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            output = text_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            values = pooled.cpu().numpy().astype(np.float32)
            if values.shape != (len(texts), 384) or not np.isfinite(values).all():
                raise ValueError("MiniLM produced invalid mean-pooled text features")
            question_vectors.update(zip(texts, values, strict=True))

    videos = np.stack([video_by_media[row["media_sha256"]] for row in rows]).astype(np.float32)
    audio = np.stack(
        [
            audio_energy_features(int(row["sound_onset_sample"]), int(row["video_frame_range"][0]))
            for row in rows
        ]
    )
    text = np.stack([question_vectors[str(row["question"])] for row in rows]).astype(np.float32)
    targets = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    if videos.shape != (len(rows), 16, 768) or audio.shape != (len(rows), 16, 1):
        raise ValueError("aligned AV feature arrays have invalid shapes")
    if text.shape != (len(rows), 384) or not all(
        np.isfinite(value).all() for value in (videos, audio, text)
    ):
        raise ValueError("aligned feature arrays have invalid values or dimensions")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    feature_path = output_dir / "decision-features.npz"
    np.savez_compressed(
        feature_path,
        sample_ids=np.asarray([row["id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        scene_ids=np.asarray([row["scene_group_id"] for row in rows]),
        pair_ids=np.asarray([row["pair_id"] for row in rows]),
        media_sha256=np.asarray([row["media_sha256"] for row in rows]),
        videos=videos,
        audios=audio,
        texts=text,
        targets=targets,
    )
    report = {
        "status": "complete_clevrer_synthetic_av_timing_feature_cache",
        "dataset": f"{DATASET}@{REVISION}",
        "source_split": "official train scenes divided into train/development validation",
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
        "sample_manifest_sha256": {
            name: sha256_file(sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "train_sample_id_order_sha256": ordered_ids_sha256(train),
        "validation_sample_id_order_sha256": ordered_ids_sha256(validation),
        "train_scene_count": len({row["scene_group_id"] for row in train}),
        "validation_scene_count": len({row["scene_group_id"] for row in validation}),
        "scene_overlap": 0,
        "media_hash_overlap": 0,
        "paired_variant_count": len({row["pair_id"] for row in rows}),
        "video_features": {
            "model_id": "google/videoprism-base-f16r288",
            "revision": cache_report["model_revision"],
            "checkpoint_sha256": cache_report["checkpoint_sha256"],
            "feature_cache_sha256": cache_report["feature_cache_sha256"],
            "shape": [len(rows), 16, 768],
        },
        "audio_features": {
            "input": (
                "synthetic 880 Hz, 80 ms Hann-windowed waveform rendered from synchronized "
                "audio onset"
            ),
            "feature": (
                "normalized waveform squared energy in 16 bins aligned to 8 video frames each"
            ),
            "shape": [len(rows), 16, 1],
            "uses_target_or_collision_annotation_as_model_input": False,
        },
        "text_features": {
            "model_id": TEXT_MODEL_ID,
            "revision": TEXT_MODEL_REVISION,
            "weight_sha256": TEXT_MODEL_SHA256,
            "parameter_count": parameter_count,
            "pooling": "attention-mask mean of last hidden state",
            "unique_questions": len(unique_questions),
            "shape": [len(rows), 384],
        },
        "validation_test_or_sealed_data_opened": False,
        "feature_builder_sha256": sha256_file(Path(__file__).resolve()),
        "feature_cache_file": feature_path.name,
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_cache_bytes": feature_path.stat().st_size,
    }
    report_path = output_dir / "feature-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["feature_report_sha256"] = sha256_file(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--video-cache-dir", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_features(
        sample_dir=args.sample_dir.resolve(strict=True),
        video_cache_dir=args.video_cache_dir.resolve(strict=True),
        text_model_dir=args.text_model_dir.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
