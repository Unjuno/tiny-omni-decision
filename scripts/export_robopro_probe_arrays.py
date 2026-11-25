"""Export selected Parquet proprioception/actions to a compact NPZ sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(sample_path: Path, data_root: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite array sidecar: {output_path}")
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    receipt_path = data_root / "download-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["sample_manifest_sha256"] != hashlib.sha256(sample_bytes).hexdigest():
        raise ValueError("receipt/sample manifest hash mismatch")
    receipt_by_key = {
        (row["split"], int(row["episode_id"]), row["kind"]): row for row in receipt["files"]
    }
    episode_ids = []
    offsets = [0]
    states = []
    actions = []
    timestamps = []
    frame_indices = []
    splits = []
    for split in ("train", "development"):
        for group in sample[split]:
            for episode in group["episodes"]:
                episode_id = int(episode["episode_id"])
                path = data_root / split / "data" / f"episode_{episode_id:06d}.parquet"
                expected_hash = receipt_by_key[(split, episode_id, "parquet")]["sha256"]
                if sha256_file(path) != expected_hash:
                    raise ValueError(f"source parquet hash mismatch: {episode_id}")
                table = pq.read_table(
                    path,
                    columns=[
                        "frame_index",
                        "episode_index",
                        "timestamp",
                        "observation.state",
                        "action",
                    ],
                )
                count = int(episode["length"])
                if table.num_rows != count:
                    raise ValueError(f"source row count mismatch: {episode_id}")
                if set(map(int, table["episode_index"].to_pylist())) != {episode_id}:
                    raise ValueError(f"source episode ID mismatch: {episode_id}")
                indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int32)
                state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
                action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                stamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
                if not np.array_equal(indices, np.arange(count)):
                    raise ValueError(f"source frame index mismatch: {episode_id}")
                if state.shape != (count, 14) or action.shape != (count, 14):
                    raise ValueError(f"source state/action shape mismatch: {episode_id}")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"non-finite source array: {episode_id}")
                episode_ids.append(episode_id)
                splits.append(split)
                states.append(state)
                actions.append(action)
                timestamps.append(stamp)
                frame_indices.append(indices)
                offsets.append(offsets[-1] + count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".partial")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            episode_ids=np.asarray(episode_ids, dtype=np.int32),
            split=np.asarray(splits),
            offsets=np.asarray(offsets, dtype=np.int64),
            states=np.concatenate(states).astype(np.float32),
            actions=np.concatenate(actions).astype(np.float32),
            timestamps=np.concatenate(timestamps).astype(np.float64),
            frame_indices=np.concatenate(frame_indices).astype(np.int32),
        )
    tmp_path.replace(output_path)
    report = {
        "schema_version": "robopro-probe-array-sidecar-v1",
        "sample_manifest_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "download_receipt_sha256": sha256_file(receipt_path),
        "episode_count": len(episode_ids),
        "total_rows": int(offsets[-1]),
        "state_shape": [int(offsets[-1]), 14],
        "action_shape": [int(offsets[-1]), 14],
        "array_sidecar": output_path.name,
        "array_sidecar_bytes": output_path.stat().st_size,
        "array_sidecar_sha256": sha256_file(output_path),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export(
        args.sample_manifest.resolve(strict=True),
        args.data_root.resolve(strict=True),
        args.output.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
