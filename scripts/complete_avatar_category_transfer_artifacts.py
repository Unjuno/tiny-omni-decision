"""Reconstruct and hash fit-only normalization for a completed AVATAR probe.

This supplements an existing run without fitting or evaluating a model.
Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_runner = importlib.import_module("scripts.run_avatar_category_transfer")
_arrays = _runner._arrays
_load_train_features = _runner._load_train_features
_query_and_video_pools = _runner._query_and_video_pools
_read_jsonl = _runner._read_jsonl
_sha256 = _runner._sha256
build_choices = _runner.build_choices
build_transfer_partition = _runner.build_transfer_partition
validate_choice_splits = _runner.validate_choice_splits


def fit_normalization(train_audio: np.ndarray, train_video: np.ndarray) -> dict[str, np.ndarray]:
    """Match the original runner's float32 fit-only means and floored stds."""
    if train_audio.ndim != 2 or train_audio.shape[1] != 384:
        raise ValueError("fit audio must have shape [N, 384]")
    if train_video.ndim != 3 or train_video.shape[1:] != (4, 768):
        raise ValueError("fit video must have shape [N, 4, 768]")
    if train_audio.shape[0] != train_video.shape[0] or not len(train_audio):
        raise ValueError("fit audio/video rows are empty or misaligned")
    if not np.isfinite(train_audio).all() or not np.isfinite(train_video).all():
        raise ValueError("fit features contain non-finite values")
    return {
        "audio_mean": train_audio.mean(axis=0),
        "audio_std": train_audio.std(axis=0).clip(1e-5),
        "video_mean": train_video.reshape(-1, 768).mean(axis=0),
        "video_std": train_video.reshape(-1, 768).std(axis=0).clip(1e-5),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def reconstruct_supplement(
    *,
    run_dir: Path,
    train_input: Path,
    train_sidecar: Path,
    audio_cache: Path,
    video_cache: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Rebuild exactly the original fit choice IDs and normalization only."""
    config_path = run_dir / "effective-config.json"
    report_path = run_dir / "report.json"
    split_path = run_dir / "split-manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frozen_split = json.loads(split_path.read_text(encoding="utf-8"))

    if _sha256(config_path) != report["effective_config_sha256"]:
        raise ValueError("effective config hash differs from the frozen report")
    input_paths = {
        "train_input_sha256": (train_input, "train_input"),
        "train_sidecar_sha256": (train_sidecar, "train_sidecar"),
        "audio_feature_cache_sha256": (audio_cache, "audio_cache"),
        "video_feature_cache_sha256": (video_cache, "video_cache"),
    }
    input_hashes = {key: _sha256(path) for key, (path, _) in input_paths.items()}
    for key, value in input_hashes.items():
        report_key = input_paths[key][1]
        if value != config[key] or value != report["input_hashes"][report_key]:
            raise ValueError(f"input hash does not match frozen run: {key}")

    source_rows = _read_jsonl(train_input)
    sidecars = _read_jsonl(train_sidecar)
    queries, videos = _query_and_video_pools(source_rows, sidecars)
    partition = build_transfer_partition(queries, videos)
    choices = {
        "fit": build_choices(
            queries,
            videos,
            split="fit",
            allowed_groups=partition["fit_groups"],
            allowed_categories=set(partition["seen_categories"]),
        ),
        "inner_validation": build_choices(
            queries,
            videos,
            split="inner_validation",
            allowed_groups=partition["validation_groups"],
            allowed_categories=set(partition["seen_categories"]),
        ),
        "held_category_test": build_choices(
            queries,
            videos,
            split="held_category_test",
            allowed_groups=partition["held_test_groups"],
            allowed_categories=set(partition["held_categories"]),
        ),
    }
    validate_choice_splits(choices)
    if partition["held_categories"] != frozen_split["held_categories"]:
        raise ValueError("reconstructed held categories differ from frozen split")
    if partition["seen_categories"] != frozen_split["seen_categories"]:
        raise ValueError("reconstructed seen categories differ from frozen split")

    split_name_map = {
        "fit": "fit",
        "inner_validation": "inner_validation",
        "held_category_test": "held_category_test",
    }
    ids_by_split: dict[str, list[str]] = {}
    for name, rows in choices.items():
        frozen = frozen_split["splits"][split_name_map[name]]
        query_ids = [row["query_id"] for row in rows]
        query_groups = sorted({row["audio_source_group"] for row in rows})
        if query_ids != frozen["query_ids"] or query_groups != frozen["query_groups"]:
            raise ValueError(f"reconstructed split differs from frozen manifest: {name}")
        ids_by_split[name] = query_ids

    fit_rows = choices["fit"]
    required_audio = {row["audio_asset_key"] for row in fit_rows}
    required_video = {
        key for row in fit_rows for key in row["video_asset_keys"]
    }
    audio_features, _ = _load_train_features(
        audio_cache, required_audio, kind="audio", dimension=384
    )
    video_features, _ = _load_train_features(
        video_cache, required_video, kind="video", dimension=768
    )
    fit_audio, fit_video, _, _, fit_ids = _arrays(
        fit_rows, audio_features, video_features
    )
    fit_ids_hash = _sha256_bytes("\n".join(fit_ids).encode())
    if fit_ids_hash != report["data"]["fit_choice_ids_sha256"]:
        raise ValueError("fit choice ID hash differs from completed run")

    stats = fit_normalization(fit_audio, fit_video)
    buffer = io.BytesIO()
    np.savez(buffer, **stats)
    normalization_bytes = buffer.getvalue()
    metadata = {
        "supplement_version": 1,
        "reconstruction_method": (
            "Deterministically rebuilt from pinned TRAIN inputs, the frozen split manifest, "
            "and fit-choice IDs. Only fit rows enter the statistics. NumPy NPZ access may "
            "materialize the full compressed TRAIN-cache feature member in memory; no labels "
            "or predictions are read, and no official-validation or sealed-audit files are used."
        ),
        "cache_read_semantics": (
            "The selected fit IDs bound the rows used in normalization. The compressed NPZ "
            "feature member is materialized as a whole by NumPy before indexed selection, so "
            "other TRAIN-cache feature rows may also be transiently present in memory."
        ),
        "reconstructed_post hoc": True,
        "run_report_sha256": _sha256(report_path),
        "effective_config_sha256": _sha256(config_path),
        "split_manifest_sha256": _sha256(split_path),
        "input_hashes": input_hashes,
        "fit_choice_ids_sha256": fit_ids_hash,
        "fit_choice_count": len(fit_rows),
        "fit_audio_asset_count": len(required_audio),
        "fit_video_asset_count": len(required_video),
        "normalization_file": "normalization.npz",
        "normalization_sha256": _sha256_bytes(normalization_bytes),
        "normalization_arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in stats.items()
        },
        "std_floor": 1e-5,
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    return normalization_bytes, metadata


def write_supplement(run_dir: Path, normalization: bytes, metadata: dict[str, Any]) -> None:
    """Create supplement files exclusively; refuse any overwrite or partial rerun."""
    normalization_path = run_dir / "normalization.npz"
    metadata_path = run_dir / "artifact-completion.json"
    if normalization_path.exists() or metadata_path.exists():
        raise FileExistsError("supplement exists; refusing to overwrite run artifacts")
    with normalization_path.open("xb") as stream:
        stream.write(normalization)
    try:
        with metadata_path.open("xb") as stream:
            stream.write(_json_bytes(metadata))
    except Exception:
        # Keep the already-written normalization as evidence of the attempted
        # supplement; the caller must inspect it rather than silently replace it.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    args = parser.parse_args()
    normalization, metadata = reconstruct_supplement(
        run_dir=args.run_dir,
        train_input=args.train_input,
        train_sidecar=args.train_sidecar,
        audio_cache=args.audio_cache,
        video_cache=args.video_cache,
    )
    write_supplement(args.run_dir, normalization, metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
