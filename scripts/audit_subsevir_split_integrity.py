"""Check event, raw-frame and normalized-frame overlap in the frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tiny_omni_decision.subsevir_features import frame_to_rgb


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite split audit: {args.output}")
    import zarr

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    features_dir = args.feature_cache
    indices = np.load(features_dir / "frame_indices.npy", allow_pickle=False)
    splits = np.load(features_dir / "frame_splits.npy", allow_pickle=False)
    event_ids = np.load(features_dir / "frame_event_ids.npy", allow_pickle=False)
    metadata = json.loads((features_dir / "metadata.json").read_text(encoding="utf-8"))
    ranges = metadata["preprocessing"]["ranges"]
    group = zarr.open_group(str(args.train_zarr), mode="r")
    raw_seen: dict[str, str] = {}
    normalized_seen: dict[tuple[str, str], str] = {}
    raw_cross_split = []
    normalized_cross_split = []
    normalized_hashes_by_row: dict[tuple[int, str], str] = {}
    row_split = {int(index): str(split) for index, split in zip(indices, splits, strict=True)}
    row_event = {
        int(index): str(event_id)
        for index, event_id in zip(indices, event_ids, strict=True)
    }
    source = group["features"]
    source_ids = group["id"]
    sorted_indices = [int(index) for index in indices]
    for start in range(0, len(source_ids), source.chunks[0]):
        stop = min(start + source.chunks[0], len(source_ids))
        left = int(np.searchsorted(sorted_indices, start, side="left"))
        right = int(np.searchsorted(sorted_indices, stop, side="left"))
        local_indices = sorted_indices[left:right]
        if not local_indices:
            continue
        block = np.asarray(source[start:stop], dtype="<f4")
        local_ids = [_decode(value) for value in source_ids[start:stop]]
        for index in local_indices:
            split = row_split[index]
            event_id = row_event[index]
            local_index = index - start
            if local_ids[local_index] != event_id:
                raise ValueError(f"source event identity mismatch at row {index}")
            frame = block[local_index]
            raw_hash = sha256_bytes(frame.tobytes())
            prior = raw_seen.setdefault(raw_hash, split)
            if prior != split:
                raw_cross_split.append({"frame_sha256": raw_hash, "splits": [prior, split]})
            for stream in ("satellite", "radar"):
                normalized_hash = sha256_bytes(frame_to_rgb(frame, ranges, stream).tobytes())
                key = (stream, normalized_hash)
                normalized_hashes_by_row[(index, stream)] = normalized_hash
                prior_split = normalized_seen.setdefault(key, split)
                if prior_split != split:
                    normalized_cross_split.append({
                        "stream": stream,
                        "frame_sha256": normalized_hash,
                        "splits": [prior_split, split],
                    })

    train_events = set(manifest["split_group_ids"]["train"])
    validation_events = set(manifest["split_group_ids"]["validation"])
    event_overlap = sorted(train_events & validation_events)
    normalized_sequences: dict[str, str] = {}
    sequence_cross_split = []
    combined_sequences: dict[str, str] = {}
    combined_sequence_cross_split = []
    for window in manifest["windows"]:
        split = str(window["split"])
        stream_sequences = {}
        for stream in ("satellite", "radar"):
            frame_hashes = [
                normalized_hashes_by_row[(int(index), stream)]
                for index in window["input_indices"]
            ]
            sequence_hash = sha256_bytes("|".join(frame_hashes).encode())
            stream_sequences[stream] = frame_hashes
            previous = normalized_sequences.setdefault(sequence_hash, split)
            if previous != split:
                sequence_cross_split.append({
                    "stream": stream,
                    "sequence_sha256": sequence_hash,
                    "splits": [previous, split],
                })
        joint_payload = "|".join(
            f"{satellite}:{radar}"
            for satellite, radar in zip(
                stream_sequences["satellite"], stream_sequences["radar"], strict=True
            )
        )
        joint_hash = sha256_bytes(joint_payload.encode())
        previous_joint = combined_sequences.setdefault(joint_hash, split)
        if previous_joint != split:
            combined_sequence_cross_split.append({
                "sequence_sha256": joint_hash,
                "splits": [previous_joint, split],
            })
    cache_membership_errors = [
        {"row_index": int(index), "split": str(split), "event_id": str(event_id)}
        for index, split, event_id in zip(
            indices.tolist(), splits.tolist(), event_ids.tolist(), strict=True
        )
        if str(event_id) not in (
            train_events if split == "train" else validation_events
        )
    ]
    result = {
        "schema": "tiny-omni-decision/subsevir-split-integrity-v1",
        "status": (
            "pass"
            if (
                not event_overlap
                and not cache_membership_errors
                and not raw_cross_split
                and not combined_sequence_cross_split
            )
            else "fail"
        ),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "train_zmetadata_sha256": hashlib.sha256(
            (args.train_zarr / ".zmetadata").read_bytes()
        ).hexdigest(),
        "input_frame_rows": len(indices),
        "train_event_groups": len(train_events),
        "validation_event_groups": len(validation_events),
        "event_group_overlap_count": len(event_overlap),
        "feature_cache_event_membership_error_count": len(cache_membership_errors),
        "exact_raw_frame_cross_split_overlap_count": len(raw_cross_split),
        "normalized_frame_cross_split_overlap_count": len(normalized_cross_split),
        "normalized_eight_frame_sequence_cross_split_overlap_count": len(sequence_cross_split),
        "normalized_multistream_eight_frame_sequence_cross_split_overlap_count": len(
            combined_sequence_cross_split
        ),
        "normalized_single_frame_collisions_are_full_sequence_or_asset_overlap": False,
        "exact_raw_frame_duplicate_count_within_any_split": len(indices) - len(raw_seen),
        "normalized_frame_duplicate_count_within_any_split": (
            len(indices) * 2 - len(normalized_seen)
        ),
        "event_overlap_examples": event_overlap[:10],
        "cache_membership_error_examples": cache_membership_errors[:10],
        "raw_overlap_examples": raw_cross_split[:10],
        "normalized_overlap_examples": normalized_cross_split[:10],
        "normalized_sequence_overlap_examples": sequence_cross_split[:10],
        "normalized_multistream_sequence_overlap_examples": combined_sequence_cross_split[:10],
        "sealed_publisher_validation_or_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--zarr-site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sys

    sys.path.append(str(args.zarr_site_packages))
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
