"""Screen fixed Physion++ TRAIN-only matched pairs for low-level render cues.

This is a descriptive data-quality audit, not a model-quality benchmark. It
opens only the preselected train pairs and never reads validation or test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

FRAME_SAMPLES = 8
SMALL_SIZE = (32, 32)
PERMUTATIONS = 20_000
BOOTSTRAPS = 20_000
SEED = 20261128
FEATURE_NAMES = (
    "file_bytes",
    "num_frames",
    "duration_seconds",
    "fps",
    "mean_red",
    "mean_green",
    "mean_blue",
    "luma_std",
    "clipped_pixel_fraction",
    "mean_adjacent_sample_luma_delta",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_indices(frame_count: int, requested: int = FRAME_SAMPLES) -> list[int]:
    if frame_count < 2 or requested < 2:
        raise ValueError("at least two source and requested frames are required")
    return sorted({round(i * (frame_count - 1) / (requested - 1)) for i in range(requested)})


def frame_pixel_features(rgb_frames: list[np.ndarray]) -> dict[str, float]:
    if len(rgb_frames) < 2:
        raise ValueError("at least two sampled RGB frames are required")
    frames = np.stack([np.asarray(frame, dtype=np.uint8) for frame in rgb_frames])
    if frames.ndim != 4 or frames.shape[-1] != 3 or not np.isfinite(frames).all():
        raise ValueError("sampled frames must be finite HxWx3 RGB arrays")
    values = frames.astype(np.float32) / 255.0
    luma = values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722
    adjacent_delta = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    return {
        "mean_red": float(values[..., 0].mean()),
        "mean_green": float(values[..., 1].mean()),
        "mean_blue": float(values[..., 2].mean()),
        "luma_std": float(luma.std()),
        "clipped_pixel_fraction": float(
            ((values <= 10 / 255) | (values >= 245 / 255)).any(axis=-1).mean()
        ),
        "mean_adjacent_sample_luma_delta": float(adjacent_delta.mean()),
    }


def paired_permutation_summary(
    differences: list[float], *, seed: int = SEED
) -> dict[str, float | int]:
    if not differences or not np.isfinite(differences).all():
        raise ValueError("paired differences must be non-empty and finite")
    values = np.asarray(differences, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(PERMUTATIONS, len(values)))
    permuted = (signs * values).mean(axis=1)
    p_value = (int(np.count_nonzero(np.abs(permuted) >= abs(observed))) + 1) / (PERMUTATIONS + 1)
    boot = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n_pairs": len(values),
        "mean_yes_minus_no": observed,
        "median_yes_minus_no": float(np.median(values)),
        "standardized_mean_difference": observed / sd if sd > 0 else 0.0,
        "bootstrap_95pct_ci_mean": [float(low), float(high)],
        "paired_sign_flip_p_value": float(p_value),
        "permutation_count": PERMUTATIONS,
        "bootstrap_count": BOOTSTRAPS,
        "seed": seed,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _safe_media_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe media path: {value}")
    return path


def _extract_video_features(row: dict[str, Any], root: Path) -> dict[str, float]:
    import av
    from PIL import Image

    media = row["media"][0]
    path = root.joinpath(*_safe_media_path(media["path"]).parts)
    if not path.is_file() or sha256_file(path) != media["sha256"]:
        raise ValueError(f"missing or hash-mismatched train video: {row['id']}")
    declared_frames = int(media["num_frames"])
    indices = set(uniform_indices(declared_frames))
    sampled: list[np.ndarray] = []
    decoded = 0
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream: {row['id']}")
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for index, frame in enumerate(container.decode(stream)):
            decoded = index + 1
            if index in indices:
                image = frame.to_image().convert("RGB")
                image.thumbnail(SMALL_SIZE, Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", SMALL_SIZE, (0, 0, 0))
                canvas.paste(
                    image, ((SMALL_SIZE[0] - image.width) // 2, (SMALL_SIZE[1] - image.height) // 2)
                )
                sampled.append(np.asarray(canvas))
    if decoded != declared_frames or len(sampled) != len(indices):
        raise ValueError(
            f"decoded frame count/sample mismatch for {row['id']}: {decoded}/{declared_frames}"
        )
    pixel = frame_pixel_features(sampled)
    return {
        "file_bytes": float(path.stat().st_size),
        "num_frames": float(decoded),
        "duration_seconds": float(decoded / fps) if fps > 0 else 0.0,
        "fps": fps,
        **pixel,
    }


def audit(train_path: Path, pairs_path: Path, corpus_root: Path) -> dict[str, Any]:
    train = _read_jsonl(train_path)
    pairs = _read_jsonl(pairs_path)
    if any(row.get("split") != "train" for row in train):
        raise ValueError("audit refuses non-train records")
    by_id = {row["id"]: row for row in train}
    if len(by_id) != len(train):
        raise ValueError("duplicate train IDs")
    if len(pairs) != 75:
        raise ValueError(f"expected the fixed 75-pair train review set, got {len(pairs)}")

    observations: list[dict[str, Any]] = []
    for pair in pairs:
        resolved: dict[str, dict[str, Any]] = {}
        for label in ("No", "Yes"):
            sample_id = pair[f"{label.lower()}_sample_id"]
            row = by_id.get(sample_id)
            if row is None or row.get("target") != label:
                raise ValueError(f"pair row is absent or target-mismatched: {sample_id}")
            if row.get("source_asset_group_id") is None:
                raise ValueError(f"missing source asset group: {sample_id}")
            resolved[label] = row
        no, yes = resolved["No"], resolved["Yes"]
        if no["source_asset_group_id"] == yes["source_asset_group_id"]:
            raise ValueError("matched pair unexpectedly shares the same source asset identity")
        if no["source_record_id"].split("#", 1)[0] != yes["source_record_id"].split("#", 1)[0]:
            raise ValueError("matched pair base scenario/factor identity differs")
        no_features = _extract_video_features(no, corpus_root)
        yes_features = _extract_video_features(yes, corpus_root)
        observations.append(
            {
                "pair_id": pair["pair_id"],
                "scenario": pair["scenario"],
                "no_sample_id": no["id"],
                "yes_sample_id": yes["id"],
                "no": no_features,
                "yes": yes_features,
            }
        )

    summaries = {}
    for name in FEATURE_NAMES:
        summaries[name] = paired_permutation_summary(
            [item["yes"][name] - item["no"][name] for item in observations]
        )
    # Holm step-down correction across the fixed, predeclared feature family.
    ordered = sorted((value["paired_sign_flip_p_value"], name) for name, value in summaries.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (p_value, name) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_value))
        adjusted[name] = running
    for name, value in summaries.items():
        value["holm_adjusted_p_value"] = adjusted[name]

    return {
        "schema_version": 1,
        "status": "complete_train_only_matched_pair_low_level_cue_screen",
        "claim_scope": (
            "descriptive screen of low-level rendering statistics; not model quality, "
            "leakage proof, or an independent benchmark"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "data_scope": (
            "only 652 existing TRAIN rows and the fixed 75 exact-recorded-factor "
            "matched No/Yes pairs; no validation/test/sealed rows"
        ),
        "train_manifest_sha256": sha256_file(train_path),
        "pair_manifest_sha256": sha256_file(pairs_path),
        "train_rows": len(train),
        "matched_pairs": len(observations),
        "source_asset_overlap_within_pairs": 0,
        "frame_sampling": {
            "count": FRAME_SAMPLES,
            "selection": "uniform endpoint-inclusive indices from manifest frame count",
            "resize": "aspect-preserving thumbnail on 32x32 black canvas",
        },
        "feature_definitions": {
            "file_bytes": "compressed media bytes",
            "num_frames": "decoded frame count, verified against manifest",
            "duration_seconds": "decoded frame count divided by stream average FPS",
            "fps": "container average frame rate",
            "mean_red_green_blue": (
                "channel means across 8 sampled 32x32 RGB frames, scaled to [0,1]"
            ),
            "luma_std": "standard deviation of BT.709 luma across sampled pixels/frames",
            "clipped_pixel_fraction": (
                "fraction of pixels with any RGB channel <=10/255 or >=245/255"
            ),
            "mean_adjacent_sample_luma_delta": (
                "mean absolute luma difference between adjacent sampled frames"
            ),
        },
        "inference": {
            "paired_test": (
                "two-sided paired sign-flip permutation test on mean Yes-minus-No difference"
            ),
            "permutations_per_feature": PERMUTATIONS,
            "bootstrap_ci": f"paired-pair bootstrap percentile 95% CI; {BOOTSTRAPS} replicates",
            "multiple_testing": "Holm correction over the 10 predeclared features",
            "seed": SEED,
        },
        "paired_feature_results": summaries,
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    report = audit(
        args.train.resolve(strict=True),
        args.pairs.resolve(strict=True),
        args.corpus_root.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "matched_pairs": report["matched_pairs"],
                "paired_feature_results": report["paired_feature_results"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
