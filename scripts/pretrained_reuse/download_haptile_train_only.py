"""Download and verify only the pinned HapTile train-task MCAPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen

REVISION = "c8593da98d3facb7a9293842514f0ee4ad3f2318"
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pretrained_reuse.freeze_haptile_task_split import (  # noqa: E402
    parse_task_counts,
)

API_ROOT = f"https://huggingface.co/api/datasets/Voxel51/HapTile/tree/{REVISION}"
DOWNLOAD_ROOT = f"https://huggingface.co/datasets/Voxel51/HapTile/resolve/{REVISION}"
DEFAULT_MAX_BYTES = 10 * 1024**3
RESERVE_BYTES = 2 * 1024**3


def _request_bytes(url: str, timeout: int = 180) -> bytes:
    request = Request(url, headers={"User-Agent": "tiny-omni-decision-research/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def _tree_page(url: str) -> tuple[bytes, str | None]:
    request = Request(url, headers={"User-Agent": "tiny-omni-decision-research/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get("Link")


def _next_tree_url(link_header: str | None, expected_path: str) -> str | None:
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link_header)
    if not match:
        if 'rel="next"' in link_header:
            raise ValueError("could not parse pinned source pagination link")
        return None
    next_url = match.group(1)
    parsed = urlsplit(next_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or unquote(parsed.path) != expected_path
    ):
        raise ValueError("pagination link escaped the pinned HapTile task tree")
    return next_url


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_tree_bytes(task: str, tree: list[dict[str, object]]) -> bytes:
    ordered = sorted(tree, key=lambda row: (str(row.get("path", "")), str(row.get("type", ""))))
    payload = {
        "format": "huggingface_tree_metadata_v1",
        "revision": REVISION,
        "task": task,
        "entries": ordered,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_task_files(
    tree: list[dict[str, object]], task: str, expected_count: int
) -> list[dict[str, object]]:
    files = [
        row for row in tree
        if row.get("type") == "file"
        and isinstance(row.get("path"), str)
        and str(row["path"]).endswith(".fo.mcap")
    ]
    if len(files) != expected_count:
        raise ValueError(
            f"{task}: pinned API has {len(files)} MCAP files, expected {expected_count}"
        )
    prefix = f"data/{task}/"
    for item in files:
        rel = str(item["path"])
        lfs = item.get("lfs")
        git_oid = item.get("oid")
        if (
            not rel.startswith(prefix)
            or ".." in Path(rel).parts
            or not isinstance(item.get("size"), int)
            or int(item["size"]) <= 0
            or not isinstance(git_oid, str)
            or len(git_oid) != 40
            or (lfs is not None and not isinstance(lfs, dict))
            or (
                isinstance(lfs, dict)
                and (
                    lfs.get("size") != item.get("size")
                    or not isinstance(lfs.get("oid"), str)
                    or len(str(lfs["oid"])) != 64
                )
            )
        ):
            raise ValueError(f"{task}: unsafe or incomplete MCAP metadata")
    return sorted(files, key=lambda row: str(row["path"]))


def _fetch_task_tree(
    task: str,
) -> tuple[str, list[bytes], list[dict[str, object]]]:
    url = f"{API_ROOT}/data/{quote(task, safe='_')}?recursive=true&expand=true"
    expected_path = unquote(urlsplit(url).path)
    pages: list[bytes] = []
    tree: list[dict[str, object]] = []
    while url:
        payload, link_header = _tree_page(url)
        page = json.loads(payload)
        if not isinstance(page, list) or not all(isinstance(row, dict) for row in page):
            raise ValueError(f"unexpected API tree response for {task}")
        pages.append(payload)
        tree.extend(page)
        url = _next_tree_url(link_header, expected_path)
    return task, pages, tree


def _download_one(item: dict[str, object], output: Path) -> dict[str, object]:
    rel = str(item["path"])
    destination = output / Path(rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lfs = item.get("lfs")
    expected_sha = str(lfs["oid"]) if isinstance(lfs, dict) else None
    expected_git_oid = str(item["oid"])
    expected_size = int(item["size"])
    if destination.exists():
        payload = destination.read_bytes()
        actual_sha = _sha256_bytes(payload)
        git_oid = hashlib.sha1(  # noqa: S324 - verifies the publisher's Git object ID
            f"blob {len(payload)}\0".encode() + payload
        ).hexdigest()
        if (
            destination.stat().st_size != expected_size
            or (expected_sha is not None and actual_sha != expected_sha)
            or (expected_sha is None and git_oid != expected_git_oid)
        ):
            raise ValueError(f"existing train sample fails pinned hash: {rel}")
    else:
        url = f"{DOWNLOAD_ROOT}/{quote(rel, safe='/')}?download=true"
        payload = _request_bytes(url)
        actual_sha = _sha256_bytes(payload)
        git_oid = hashlib.sha1(  # noqa: S324 - verifies the publisher's Git object ID
            f"blob {len(payload)}\0".encode() + payload
        ).hexdigest()
        if (
            len(payload) != expected_size
            or (expected_sha is not None and actual_sha != expected_sha)
            or (expected_sha is None and git_oid != expected_git_oid)
        ):
            raise ValueError(f"download failed pinned size/SHA-256: {rel}")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".haptile-", suffix=".part",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return {
        "path": rel,
        "size": expected_size,
        "sha256": actual_sha,
        "git_blob_oid": expected_git_oid,
        "lfs_sha256": expected_sha,
    }


def download_train_only(
    *,
    card: Path,
    split_manifest: Path,
    output: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    api_workers: int = 6,
    download_workers: int = 8,
    resume: bool = False,
) -> dict[str, object]:
    if max_bytes <= 0 or api_workers < 1 or download_workers < 1:
        raise ValueError("size and worker limits must be positive")
    resolved_output = output.resolve()
    if resolved_output == REPO_ROOT or REPO_ROOT in resolved_output.parents:
        raise ValueError("media must be stored outside the Git worktree")
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if manifest.get("conversion_revision") != REVISION:
        raise ValueError("public split manifest differs from the pinned conversion")
    splits = manifest.get("development_splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("train"), list):
        raise ValueError("public split manifest lacks train task groups")
    train_tasks = list(splits["train"])
    heldout_tasks = set(splits.get("calibration", [])) | set(splits.get("validation", []))
    if len(train_tasks) != 24 or set(train_tasks) & heldout_tasks:
        raise ValueError("task split is invalid or overlaps held-out development groups")
    expected_counts, card_hash = parse_task_counts(card)
    if card_hash != manifest.get("conversion_card_sha256"):
        raise ValueError("task card hash differs from the public split manifest")
    if sum(expected_counts.get(task, 0) for task in train_tasks) != 1113:
        raise ValueError("pinned train tasks do not sum to 1,113 episodes")

    if resolved_output.exists() and not resume:
        raise FileExistsError("output exists; pass --resume to verify and continue it")
    resolved_output.mkdir(parents=True, exist_ok=True)
    metadata_dir = resolved_output / "metadata-canonical"
    metadata_dir.mkdir(exist_ok=True)

    task_trees: dict[str, tuple[list[bytes], list[dict[str, object]]]] = {}
    with ThreadPoolExecutor(max_workers=api_workers) as pool:
        futures = {pool.submit(_fetch_task_tree, task): task for task in train_tasks}
        for future in as_completed(futures):
            task, pages, tree = future.result()
            task_trees[task] = (pages, tree)
    selected: list[dict[str, object]] = []
    tree_receipts: dict[str, dict[str, object]] = {}
    for task in sorted(train_tasks):
        pages, tree = task_trees[task]
        entries = select_task_files(tree, task, expected_counts[task])
        selected.extend(entries)
        metadata_payload = _canonical_tree_bytes(task, tree)
        tree_receipts[task] = {
            "canonical_sha256": _sha256_bytes(metadata_payload),
            "api_pages": len(pages),
            "api_entry_count": len(tree),
            "mcap_count": len(entries),
        }
        metadata_path = metadata_dir / f"{task}.json"
        if metadata_path.exists():
            existing_hash = _sha256_file(metadata_path)
            expected_hash = _sha256_bytes(metadata_payload)
            if existing_hash != expected_hash:
                raise ValueError(
                    f"existing pinned metadata differs: {task} "
                    f"({existing_hash} != {expected_hash})"
                )
        else:
            metadata_path.write_bytes(metadata_payload)

    total_bytes = sum(int(item["size"]) for item in selected)
    if len(selected) != 1113 or total_bytes > max_bytes:
        raise ValueError(
            f"train corpus is {len(selected)} files / {total_bytes} bytes; "
            f"limit is {max_bytes} bytes"
        )
    free_bytes = shutil.disk_usage(resolved_output.parent).free
    existing_bytes = sum(
        path.stat().st_size for path in (resolved_output / "data").rglob("*.fo.mcap")
    ) if (resolved_output / "data").exists() else 0
    if free_bytes < (total_bytes - existing_bytes + RESERVE_BYTES):
        raise OSError("insufficient disk space after preserving the 2 GiB reserve")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(_download_one, item, resolved_output): item for item in selected}
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 50 == 0:
                print(f"verified {index}/{len(selected)} train MCAPs", flush=True)
    results.sort(key=lambda row: str(row["path"]))
    receipt = {
        "format": "haptile_frozen_train_corpus_v1",
        "source_revision": manifest["source_revision"],
        "conversion_revision": REVISION,
        "conversion_card_sha256": card_hash,
        "split_manifest_file_sha256": _sha256_file(split_manifest),
        "split_manifest_commitment_sha256": manifest["manifest_sha256"],
        "accessed_splits": ["train"],
        "task_count": len(train_tasks),
        "episode_count": len(results),
        "total_bytes": sum(int(row["size"]) for row in results),
        "task_tree_receipts": tree_receipts,
        "files": results,
        "validation_or_sealed_audit_opened": False,
    }
    receipt_path = resolved_output / "train-corpus-receipt.json"
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if receipt_path.read_bytes() != payload:
            raise ValueError("existing train corpus receipt differs; refusing overwrite")
    else:
        temp = receipt_path.with_suffix(".json.part")
        temp.write_bytes(payload)
        os.rename(temp, receipt_path)
    print(json.dumps({
        "status": "train-only corpus verified",
        "episodes": receipt["episode_count"],
        "bytes": receipt["total_bytes"],
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256_bytes(payload),
    }, indent=2), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--api-workers", type=int, default=6)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    download_train_only(
        card=args.card,
        split_manifest=args.split_manifest,
        output=args.output,
        max_bytes=args.max_bytes,
        api_workers=args.api_workers,
        download_workers=args.download_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
