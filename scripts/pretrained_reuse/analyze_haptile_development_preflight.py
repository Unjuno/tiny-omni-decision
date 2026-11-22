"""Create action targets from one frozen HapTile development split only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.haptile_action import (  # noqa: E402
    ACTION_LABELS,
    DEFAULT_MOTION_DEADBAND_M,
    directional_action_label,
)

PINNED_REVISION = "c8593da98d3facb7a9293842514f0ee4ad3f2318"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_development_receipt(
    receipt: dict[str, object], manifest: dict[str, object], manifest_sha256: str,
    split_name: str,
) -> list[dict[str, object]]:
    """Fail closed unless receipt is exactly the requested nonsealed split."""
    if split_name not in {"validation", "calibration"}:
        raise ValueError("only validation or calibration may be analyzed")
    if receipt.get("format") != "haptile_development_split_corpus_v1":
        raise ValueError("input is not a frozen HapTile development corpus")
    if receipt.get("source_revision") != "54887384f1fe9e93b799e104194a582f9d8840b8":
        raise ValueError("HapTile source revision differs from the pinned release")
    if receipt.get("conversion_revision") != PINNED_REVISION:
        raise ValueError("HapTile conversion revision differs from the pinned release")
    if receipt.get("split_manifest_file_sha256") != manifest_sha256:
        raise ValueError("receipt uses a different frozen split manifest")
    if receipt.get("accessed_splits") != [split_name]:
        raise ValueError("receipt includes a different or multiple data split")
    if receipt.get("sealed_audit_opened") is not False:
        raise ValueError("receipt does not prove sealed audit isolation")
    if manifest.get("sealed_audit_accessed") is not False:
        raise ValueError("public manifest does not prove sealed audit isolation")
    splits = manifest.get("development_splits")
    totals = manifest.get("development_episode_counts")
    if not isinstance(splits, dict) or not isinstance(totals, dict):
        raise ValueError("public manifest lacks frozen development split commitments")
    requested = splits.get(split_name)
    train = splits.get("train")
    sealed = manifest.get("sealed_audit", {})
    sealed_count = sealed.get("task_group_count") if isinstance(sealed, dict) else None
    if not isinstance(requested, list) or not isinstance(train, list):
        raise ValueError("public manifest lacks the requested development groups")
    if set(requested) & set(train):
        raise ValueError("requested development groups overlap train")
    # The public manifest intentionally has no sealed task names. Receipt task
    # identity is additionally checked against the one requested split below.
    if sealed_count is None:
        raise ValueError("public manifest lacks sealed split commitment")
    files = receipt.get("files")
    expected_count = totals.get(split_name)
    if (not isinstance(files, list) or not files or not isinstance(expected_count, int)
            or len(files) != expected_count or receipt.get("episode_count") != expected_count):
        raise ValueError("development receipt does not match the frozen episode count")
    paths: set[str] = set()
    tasks: set[str] = set()
    counts_by_task: Counter[str] = Counter()
    total_bytes = 0
    validated: list[dict[str, object]] = []
    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError("receipt contains a malformed file row")
        rel = Path(str(raw.get("path", "")))
        task = raw.get("task", rel.parts[1] if len(rel.parts) == 3 else None)
        if (not isinstance(task, str) or task not in requested or rel.is_absolute()
                or ".." in rel.parts or len(rel.parts) != 3 or rel.parts[0] != "data"
                or rel.parts[1] != task or rel.suffix != ".mcap"):
            raise ValueError("receipt contains a nonrequested task or unsafe media path")
        normalized = rel.as_posix()
        if normalized in paths:
            raise ValueError("receipt contains duplicate media paths")
        paths.add(normalized)
        tasks.add(task)
        counts_by_task[task] += 1
        size = raw.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("receipt contains an invalid media byte size")
        total_bytes += size
        validated.append(raw)
    if tasks != set(requested):
        raise ValueError("receipt does not cover exactly the requested task groups")
    if receipt.get("task_count") != len(requested):
        raise ValueError("receipt task count differs from the frozen split")
    trees = receipt.get("task_tree_receipts")
    if not isinstance(trees, dict) or set(trees) != set(requested):
        raise ValueError("task metadata receipts do not match the requested split")
    for task in requested:
        row = trees.get(task)
        if not isinstance(row, dict) or row.get("mcap_count") != counts_by_task[task]:
            raise ValueError("task media count differs from its pinned metadata receipt")
    if receipt.get("total_bytes") != total_bytes:
        raise ValueError("receipt total byte count differs from its media inventory")
    return validated


def analyze_development_receipt(
    receipt_path: Path, output_path: Path, manifest_path: Path, split_name: str,
) -> dict[str, object]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError("install the pinned MCAP decoder dependencies") from exc
    if split_name not in {"validation", "calibration"}:
        raise ValueError("only validation or calibration may be analyzed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    files = validate_development_receipt(receipt, manifest, manifest_hash, split_name)
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    total_messages = 0
    for item in files:
        rel = Path(str(item["path"]))
        task = str(item.get("task", rel.parts[1]))
        path = receipt_path.parent.joinpath(*rel.parts)
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise ValueError(f"missing or size-mismatched development sample: {rel}")
        digest = sha256_file(path)
        if digest != item.get("sha256"):
            raise ValueError(f"development sample SHA-256 mismatch: {rel}")
        if item.get("lfs_sha256") is not None:
            if digest != item["lfs_sha256"]:
                raise ValueError(f"development LFS SHA-256 mismatch: {rel}")
        elif git_blob_sha1(path) != item.get("git_blob_oid"):
            raise ValueError(f"development Git blob OID mismatch: {rel}")
        times: list[float] = []
        positions: list[tuple[float, float, float]] = []
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()], validate_crcs=True)
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=["/end-effector-pose"]
            ):
                if schema is None or schema.name != "foxglove.PoseInFrame":
                    raise ValueError(f"unexpected pose schema for {rel}")
                if channel.topic != "/end-effector-pose":
                    raise ValueError(f"unexpected pose topic for {rel}")
                p = decoded.pose.position
                times.append(message.log_time / 1_000_000_000)
                positions.append((p.x, p.y, p.z))
        if len(times) < 4 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError(f"insufficient or nonmonotonic pose stream: {rel}")
        query = (times[0] + times[-1]) / 2
        paired = zip(times, positions, strict=True)
        current = [p for t, p in paired if query - 0.25 <= t <= query]
        paired = zip(times, positions, strict=True)
        future = [p for t, p in paired if query + 0.25 <= t <= query + 0.5]
        if not current or not future:
            raise ValueError(f"midpoint lacks the fixed target window: {rel}")
        current_xyz = tuple(statistics.median(p[a] for p in current) for a in range(3))
        future_xyz = tuple(statistics.median(p[a] for p in future) for a in range(3))
        target = directional_action_label(current_xyz, future_xyz)
        counts[target] += 1
        total_messages += len(times)
        rows.append({
            "task": task,
            "path": rel.as_posix(),
            "sha256": digest,
            "pose_messages": len(times),
            "query_time_log_seconds": round(query, 6),
            "current_xyz_m": current_xyz,
            "future_xyz_m": future_xyz,
            "target": target,
        })
    output = {
        "format": f"haptile_{split_name}_action_preflight_v1",
        "conversion_revision": PINNED_REVISION,
        "input_receipt_sha256": sha256_file(receipt_path),
        "split_manifest_sha256": manifest_hash,
        "split_accessed": split_name,
        "query_protocol": {
            "query": "episode midpoint; input cutoff inclusive",
            "target_current_pose_window": "[t-0.25s, t]",
            "target_future_pose_window": "[t+0.25s, t+0.50s]",
            "target": "dominant Cartesian displacement direction, else hold",
            "deadband_m": DEFAULT_MOTION_DEADBAND_M,
            "labels": list(ACTION_LABELS),
        },
        "sample_count": len(rows),
        "pose_messages": total_messages,
        "class_counts": {label: counts[label] for label in ACTION_LABELS},
        "samples": rows,
        "sealed_audit_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite action preflight: {output_path}")
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "calibration"), required=True)
    args = parser.parse_args()
    result = analyze_development_receipt(
        args.receipt, args.output, args.split_manifest, args.split
    )
    print(json.dumps({
        "split": args.split,
        "samples": result["sample_count"],
        "class_counts": result["class_counts"],
        "sealed_audit_opened": result["sealed_audit_opened"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
