"""Fetch only frozen HapTile validation or calibration task groups."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pretrained_reuse.download_haptile_train_only import (  # noqa: E402
    RESERVE_BYTES,
    _canonical_tree_bytes,
    _download_one,
    _fetch_task_tree,
    _sha256_bytes,
    _sha256_file,
    select_task_files,
)
from scripts.pretrained_reuse.freeze_haptile_task_split import (  # noqa: E402
    parse_task_counts,
)


def development_tasks(
    manifest: dict[str, object], card_counts: dict[str, int], split_name: str
) -> list[str]:
    """Allow only nonsealed validation/calibration groups, never train/audit."""
    if split_name not in {"validation", "calibration"}:
        raise ValueError("only frozen validation or calibration groups may be fetched")
    splits = manifest.get("development_splits")
    totals = manifest.get("development_episode_counts")
    if not isinstance(splits, dict) or not isinstance(totals, dict):
        raise ValueError("public split manifest lacks development group commitments")
    tasks = splits.get(split_name)
    if not isinstance(tasks, list) or not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("development split task groups are missing or duplicated")
    all_development = [
        task for name in ("train", "calibration", "validation")
        for task in splits.get(name, [])
    ]
    if len(all_development) != len(set(all_development)):
        raise ValueError("public development task groups overlap")
    if any(task not in card_counts for task in tasks):
        raise ValueError("split task is absent from the pinned conversion card")
    expected_count = sum(card_counts[task] for task in tasks)
    if expected_count != totals.get(split_name):
        raise ValueError("pinned task counts differ from the split commitment")
    return sorted(str(task) for task in tasks)


def download_development_split(
    *,
    card: Path,
    split_manifest: Path,
    output: Path,
    split_name: str,
    max_bytes: int = 10 * 1024**3,
    api_workers: int = 4,
    download_workers: int = 6,
    resume: bool = False,
) -> dict[str, object]:
    if max_bytes <= 0 or api_workers < 1 or download_workers < 1:
        raise ValueError("size and worker limits must be positive")
    resolved_output = output.resolve()
    if resolved_output == REPO_ROOT or REPO_ROOT in resolved_output.parents:
        raise ValueError("media must be stored outside the Git worktree")
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    counts, card_hash = parse_task_counts(card)
    if (card_hash != manifest.get("conversion_card_sha256")
            or manifest.get("conversion_revision") != "c8593da98d3facb7a9293842514f0ee4ad3f2318"):
        raise ValueError("pinned conversion card or source revision differs")
    tasks = development_tasks(manifest, counts, split_name)
    expected_episodes = int(manifest["development_episode_counts"][split_name])
    if resolved_output.exists() and not resume:
        raise FileExistsError("output exists; pass --resume to verify the existing attempt")
    resolved_output.mkdir(parents=True, exist_ok=True)

    task_trees: dict[str, tuple[list[bytes], list[dict[str, object]]]] = {}
    with ThreadPoolExecutor(max_workers=api_workers) as pool:
        futures = {pool.submit(_fetch_task_tree, task): task for task in tasks}
        for future in as_completed(futures):
            task, pages, tree = future.result()
            task_trees[task] = (pages, tree)

    selected: list[dict[str, object]] = []
    tree_receipts: dict[str, dict[str, object]] = {}
    metadata_dir = resolved_output / "metadata-canonical"
    metadata_dir.mkdir(exist_ok=True)
    for task in tasks:
        pages, tree = task_trees[task]
        entries = select_task_files(tree, task, counts[task])
        selected.extend(entries)
        canonical = _canonical_tree_bytes(task, tree)
        tree_receipts[task] = {
            "canonical_sha256": _sha256_bytes(canonical),
            "api_pages": len(pages),
            "api_entry_count": len(tree),
            "mcap_count": len(entries),
        }
        metadata_path = metadata_dir / f"{task}.json"
        if metadata_path.exists() and _sha256_file(metadata_path) != _sha256_bytes(canonical):
            raise ValueError(f"existing pinned metadata differs: {task}")
        if not metadata_path.exists():
            metadata_path.write_bytes(canonical)

    total_bytes = sum(int(item["size"]) for item in selected)
    if len(selected) != expected_episodes or total_bytes > max_bytes:
        raise ValueError("selected development corpus count or byte cap is invalid")
    existing_bytes = sum(
        path.stat().st_size for path in (resolved_output / "data").rglob("*.fo.mcap")
    ) if (resolved_output / "data").exists() else 0
    free_bytes = shutil.disk_usage(resolved_output.parent).free
    if free_bytes < total_bytes - existing_bytes + RESERVE_BYTES:
        raise OSError("insufficient disk space after preserving the 2 GiB reserve")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(_download_one, item, resolved_output): item for item in selected}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 50 == 0:
                print(f"verified {index}/{len(selected)} {split_name} MCAPs", flush=True)
    results.sort(key=lambda row: str(row["path"]))
    receipt = {
        "format": "haptile_development_split_corpus_v1",
        "source_revision": manifest["source_revision"],
        "conversion_revision": manifest["conversion_revision"],
        "conversion_card_sha256": card_hash,
        "split_manifest_file_sha256": _sha256_file(split_manifest),
        "split_manifest_commitment_sha256": manifest["manifest_sha256"],
        "accessed_splits": [split_name],
        "task_count": len(tasks),
        "episode_count": len(results),
        "total_bytes": sum(int(row["size"]) for row in results),
        "task_tree_receipts": tree_receipts,
        "files": results,
        "sealed_audit_opened": False,
    }
    receipt_path = resolved_output / f"{split_name}-corpus-receipt.json"
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists() and receipt_path.read_bytes() != payload:
        raise ValueError("existing development receipt differs; refusing overwrite")
    if not receipt_path.exists():
        temporary = receipt_path.with_suffix(".json.part")
        temporary.write_bytes(payload)
        temporary.replace(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "calibration"), required=True)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = download_development_split(
        card=args.card,
        split_manifest=args.split_manifest,
        output=args.output,
        split_name=args.split,
        max_bytes=args.max_bytes,
        api_workers=args.api_workers,
        download_workers=args.download_workers,
        resume=args.resume,
    )
    print(json.dumps({
        "split": args.split,
        "episodes": result["episode_count"],
        "bytes": result["total_bytes"],
        "sealed_audit_opened": result["sealed_audit_opened"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
