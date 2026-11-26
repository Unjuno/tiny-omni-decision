"""Freeze a deterministic storm-grouped sub-SEVIR Path A manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import zarr

from tiny_omni_decision.subsevir_task import build_forecast_windows, canonical_sha256, window_dict


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def freeze(train_zarr: Path, output: Path, *, seed: int = 17) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    group = zarr.open_group(str(train_zarr), mode="r")
    if "id" not in group or "n_samples" not in group or "label_1d_class" not in group:
        raise ValueError("source is missing required sub-SEVIR arrays")

    event_ids = [_decode(value) for value in group["id"][:]]
    timestamps = [int(value) for value in group["n_samples"][:]]
    labels = [int(value) for value in group["label_1d_class"][:]]
    if not (len(event_ids) == len(timestamps) == len(labels)):
        raise ValueError("source arrays have inconsistent lengths")

    windows = build_forecast_windows(event_ids, timestamps, labels, seed=seed)
    event_to_split: dict[str, str] = {}
    for window in windows:
        previous = event_to_split.setdefault(window.event_id, window.split)
        if previous != window.split:
            raise AssertionError(f"event crosses splits: {window.event_id}")

    records = [window_dict(window) for window in windows]
    split_windows = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation")
    }
    group_ids = {
        split: sorted({str(record["event_id"]) for record in split_records})
        for split, split_records in split_windows.items()
    }
    for split, split_records in split_windows.items():
        if any(str(record["split"]) != split for record in split_records):
            raise AssertionError(f"bad split membership in {split}")
    if set(group_ids["train"]) & set(group_ids["validation"]):
        raise AssertionError("storm event overlap between train and validation")

    zmeta = train_zarr / ".zmetadata"
    zmeta_sha = hashlib.sha256(zmeta.read_bytes()).hexdigest() if zmeta.exists() else None
    manifest: dict[str, object] = {
        "schema": "tiny-omni-decision/subsevir-patha-manifest-v1",
        "seed": seed,
        "split_method": "SHA-256(event_id, seed), 20% validation threshold; label-independent",
        "source": {
            "train_zarr": str(train_zarr.resolve()),
            "zmetadata_sha256": zmeta_sha,
            "rows": len(event_ids),
            "event_id_sha256": canonical_sha256(event_ids),
            "timestamps_minutes_sha256": canonical_sha256(timestamps),
            "label_1d_class_sha256": canonical_sha256(labels),
        },
        "task": {
            "description": (
                "Use 8 prior 5-minute frames to predict lightning presence "
                "in the next frame."
            ),
            "history_frames": 8,
            "cadence_minutes": 5,
            "target": "label_1d_class at immediately following frame",
            "eligible_only_if_all_9_timestamps_are_exactly_5_minutes_apart": True,
        },
        "excluded_irregular_or_short_windows": True,
        "split_group_ids": group_ids,
        "split_group_counts": {key: len(value) for key, value in group_ids.items()},
        "split_window_counts": {key: len(value) for key, value in split_windows.items()},
        "split_target_counts": {
            key: dict(sorted(Counter(int(record["target"]) for record in value).items()))
            for key, value in split_windows.items()
        },
        "windows_sha256": canonical_sha256(records),
        "windows": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = freeze(args.train_zarr, args.output, seed=args.seed)
    summary_keys = (
        "schema",
        "split_group_counts",
        "split_window_counts",
        "split_target_counts",
        "windows_sha256",
    )
    print(json.dumps({key: manifest[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
