"""Create a new, source-group-disjoint AVATAR fold using TRAIN records only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.fit_avatar_decision_readout import (  # noqa: E402
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from scripts.run_avatar_category_transfer import (  # noqa: E402
    _query_and_video_pools,
    build_choices,
    validate_choice_splits,
)  # noqa: E402

SEED = 17
TRAIN_FRACTION = 0.8


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _group_partition(groups: set[str], *, seed: int = SEED) -> tuple[set[str], set[str]]:
    if len(groups) < 10:
        raise ValueError("at least ten source groups are required")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"avatar-dinov2-train-fold-v1:{seed}:{group}".encode()
        ).hexdigest(),
    )
    fit_count = round(len(ordered) * TRAIN_FRACTION)
    fit, validation = set(ordered[:fit_count]), set(ordered[fit_count:])
    if not fit or not validation or fit & validation or fit | validation != groups:
        raise ValueError("invalid source-group partition")
    return fit, validation


def _feature_subset(source: Path, output: Path, expected: set[str], *, kind: str) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        keys = [str(value) for value in archive["asset_keys"].tolist()]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate IDs in {kind} source cache")
        index = {key: position for position, key in enumerate(keys)}
        missing = expected - index.keys()
        if missing:
            raise ValueError(f"{kind} source cache lacks {len(missing)} required TRAIN assets")
        selected = sorted(expected)
        positions = np.asarray([index[key] for key in selected], dtype=np.int64)
        arrays: dict[str, np.ndarray] = {
            "asset_keys": np.asarray(selected),
            "features": np.asarray(archive["features"][positions], dtype=np.float32),
        }
        extras = (
            ("centers_seconds", "padding_samples", "declared_padding_samples")
            if kind == "audio"
            else ("timestamps", "max_frame_time_error_seconds")
        )
        for key in extras:
            arrays[key] = np.asarray(archive[key][positions])
        if not np.isfinite(arrays["features"]).all():
            raise ValueError(f"non-finite {kind} source features")
        if kind == "audio":
            arrays["features"] = arrays["features"].astype(np.float32, copy=False)
        else:
            arrays["features"] = arrays["features"].astype(np.float32, copy=False)
        np.savez_compressed(output, **arrays)
    return {"asset_count": len(expected), "path": str(output), "sha256": sha256_file(output)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_input = args.train_input.resolve(strict=True)
    train_sidecar = args.train_sidecar.resolve(strict=True)
    audio_cache = args.audio_cache.resolve(strict=True)
    videoprism_cache = args.videoprism_cache.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")

    model_rows = read_jsonl(train_input)
    sidecar_rows = read_jsonl(train_sidecar)
    queries, videos = _query_and_video_pools(model_rows, sidecar_rows)
    groups = {str(row["source_group"]) for row in queries + videos}
    categories = {str(row["category"]) for row in queries + videos}
    fit_groups, validation_groups = _group_partition(groups)
    choices = {
        "train": build_choices(
            queries,
            videos,
            split="train",
            allowed_groups=fit_groups,
            allowed_categories=categories,
        ),
        "validation": build_choices(
            queries,
            videos,
            split="validation",
            allowed_groups=validation_groups,
            allowed_categories=categories,
        ),
    }
    integrity = validate_choice_splits(choices)
    if min(len(choices["train"]), len(choices["validation"])) < 500:
        raise ValueError("TRAIN-only group fold has insufficient choice rows")

    query_rows = {str(row["audio_asset_key"]): row for row in model_rows}
    fit_inputs: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    fit_sidecars: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, rows in choices.items():
        for choice in rows:
            source = query_rows[choice["audio_asset_key"]]
            example_id = hashlib.sha256(
                f"avatar-dinov2-fold-v1:{split}:{choice['query_id']}".encode()
            ).hexdigest()[:24]
            option_ids = [f"candidate_{index}" for index in range(4)]
            options = [
                {
                    "option_id": option_ids[index],
                    "video_asset_key": asset,
                    "video_timestamp_seconds": timestamp,
                }
                for index, (asset, timestamp) in enumerate(
                    zip(
                        choice["video_asset_keys"],
                        choice["video_timestamps_seconds"],
                        strict=True,
                    )
                )
            ]
            fit_inputs[split].append(
                {
                    "example_id": example_id,
                    "split": split,
                    "question": "Which candidate matches the audio input?",
                    "audio_asset_key": choice["audio_asset_key"],
                    "audio_window_center_seconds": source["audio_window_center_seconds"],
                    "audio_window_duration_seconds": source["audio_window_duration_seconds"],
                    "audio_window_left_padding_seconds": source[
                        "audio_window_left_padding_seconds"
                    ],
                    "audio_window_right_padding_seconds": source[
                        "audio_window_right_padding_seconds"
                    ],
                    "options": options,
                    "target_option_id": option_ids[choice["target_index"]],
                }
            )
            fit_sidecars[split].append(
                {
                    "example_id": example_id,
                    "target_option_index": choice["target_index"],
                    "option_asset_keys": choice["video_asset_keys"],
                    "option_source_groups": choice["video_source_groups"],
                    "option_categories": choice["option_categories"],
                    "target_source_group": choice["video_source_groups"][choice["target_index"]],
                    "target_category": choice["category"],
                    "audio_source_group": choice["audio_source_group"],
                }
            )

    expected_audio = {row["audio_asset_key"] for rows in fit_inputs.values() for row in rows}
    expected_video = {
        option["video_asset_key"]
        for rows in fit_inputs.values()
        for row in rows
        for option in row["options"]
    }
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        _write_jsonl(output / f"{split}-input.jsonl", fit_inputs[split])
        _write_jsonl(output / f"{split}-sidecar.jsonl", fit_sidecars[split])
    audio_subset = _feature_subset(
        audio_cache, output / "audio-features-train-only.npz", expected_audio, kind="audio"
    )
    videoprism_subset = _feature_subset(
        videoprism_cache,
        output / "videoprism-features-train-only.npz",
        expected_video,
        kind="video",
    )
    split_hashes = {
        split: hashlib.sha256(
            "\n".join(row["example_id"] for row in fit_inputs[split]).encode()
        ).hexdigest()
        for split in ("train", "validation")
    }
    report = {
        "experiment_id": "avatar-dinov2-train-only-group-fold-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "source_revision": "a7f0146ff704799d044133e61137836cacc1c233",
        "task_scope": (
            "four-option cross-source audio-video category match; not native AVATAR localization"
        ),
        "split_source": (
            "original AVATAR TRAIN rows only; official validation and sealed audit were not loaded"
        ),
        "partition": {
            "method": "SHA-256 rank of source-group identity",
            "train_group_fraction": TRAIN_FRACTION,
            "fit_source_group_count": len(fit_groups),
            "validation_source_group_count": len(validation_groups),
            "integrity": integrity,
            "example_id_order_sha256": split_hashes,
        },
        "counts": {
            "source_train_rows": len(model_rows),
            "source_train_source_groups": len(groups),
            "categories": len(categories),
            "train_choices": len(choices["train"]),
            "validation_choices": len(choices["validation"]),
            "train_audio_assets": len({r["audio_asset_key"] for r in fit_inputs["train"]}),
            "validation_audio_assets": len(
                {r["audio_asset_key"] for r in fit_inputs["validation"]}
            ),
            "unique_video_assets": len(expected_video),
        },
        "inputs": {
            "train_input_sha256": sha256_file(train_input),
            "train_sidecar_sha256": sha256_file(train_sidecar),
            "audio_cache_sha256": sha256_file(audio_cache),
            "videoprism_cache_sha256": sha256_file(videoprism_cache),
        },
        "caches": {"audio": audio_subset, "videoprism": videoprism_subset},
        "sealed_audit_accessed": False,
        "media_written_to_output": False,
    }
    write_json_atomic(output / "fold-manifest.json", report)
    report["fold_manifest_sha256"] = sha256_file(output / "fold-manifest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--train-sidecar", required=True, type=Path)
    parser.add_argument("--audio-cache", required=True, type=Path)
    parser.add_argument("--videoprism-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(build(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
