"""Fetch pinned CLEVRER reasoning clips without materializing full archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

from scripts.fetch_clevrer_probe import (
    TRAIN_BYTES,
    TRAIN_ETAG,
    TRAIN_URL,
    VALIDATION_BYTES,
    VALIDATION_ETAG,
    VALIDATION_URL,
    RangeFile,
)
from tiny_omni_decision.dataset import sha256_file

REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
SCREEN_PATH = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\clevrer-reasoning-coverage-screen-20261021-attempt-04\screen.json"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _selected_partition_names(holdout_only: bool) -> tuple[str, ...]:
    return ("holdout",) if holdout_only else ("train", "development")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _retrying_range_file(url: str, size: int, etag: str) -> RangeFile:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    reader = RangeFile(url, size, etag)
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    reader.session.mount("https://", HTTPAdapter(max_retries=retry))
    return reader


def _historical_media_hashes(screen_path: Path) -> set[str]:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    hashes: set[str] = set()
    for item in screen["exposure_audit"]["exposure_files"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"prior exposure input changed: {path}")
        for row in _read_jsonl(path):
            digest = row.get("media_sha256")
            if isinstance(digest, str) and len(digest) == 64:
                hashes.add(digest)
    return hashes


def fetch(
    *,
    preflight_dir: Path,
    output_dir: Path,
    screen_path: Path = SCREEN_PATH,
    resume: bool = False,
    holdout_only: bool = False,
    previous_sample_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite media output: {output_dir}")
    if resume and not output_dir.is_dir():
        raise FileNotFoundError(f"cannot resume missing output directory: {output_dir}")
    if resume and (output_dir / "fetch-report.json").exists():
        raise FileExistsError("fetch report already exists; refusing to resume a completed sample")
    preflight_path = preflight_dir / "preflight.json"
    report_path = json.loads(preflight_path.read_text(encoding="utf-8"))
    if report_path.get("status") != "preflight_pass_clevrer_fresh_reasoning":
        raise ValueError("fresh CLEVRER preflight is not passing")
    if hashlib.sha256(screen_path.read_bytes()).hexdigest() != report_path["exposure"][
        "screen_sha256"
    ]:
        raise ValueError("source-screen input changed after preflight")
    prior_hashes = _historical_media_hashes(screen_path)
    selected_names = _selected_partition_names(holdout_only)
    selected = {
        name: _read_jsonl(preflight_dir / f"{name}.jsonl") for name in selected_names
    }
    for name, rows in selected.items():
        expected = report_path["partitions"][name]
        ids_hash = hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()
        if ids_hash != expected["choice_id_order_sha256"]:
            raise ValueError(f"{name} selection differs from the frozen preflight")
        if hashlib.sha256(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ).encode()
        ).hexdigest() != expected["canonical_rows_sha256"]:
            raise ValueError(f"{name} row content differs from the frozen preflight")

    archives = {
        "train": (TRAIN_URL, TRAIN_BYTES, TRAIN_ETAG),
        "validation": (VALIDATION_URL, VALIDATION_BYTES, VALIDATION_ETAG),
    }
    output_dir.mkdir(parents=True, exist_ok=resume)
    media_hashes_by_partition: dict[str, set[str]] = {name: set() for name in selected}
    downloaded_by_split: dict[str, int] = {name: 0 for name in archives}
    reused_by_split: dict[str, int] = {name: 0 for name in archives}
    range_bytes_by_split: dict[str, int] = {name: 0 for name in archives}

    for source_split, (url, size, etag) in archives.items():
        scene_rows: dict[int, dict[str, Any]] = {}
        for partition, rows in selected.items():
            if source_split == "train" and partition != "train":
                continue
            if source_split == "validation" and partition == "train":
                continue
            for row in rows:
                scene_rows[int(row["scene_index"])] = row
        if not scene_rows:
            continue
        reader = _retrying_range_file(url, size, etag)
        try:
            import zipfile

            with zipfile.ZipFile(reader) as archive:
                names_by_basename: dict[str, list[str]] = {}
                for name in archive.namelist():
                    if name.lower().endswith(".mp4"):
                        names_by_basename.setdefault(Path(name).name, []).append(name)
                for scene, source_row in sorted(scene_rows.items()):
                    matches = names_by_basename.get(str(source_row["video_filename"]), [])
                    if len(matches) != 1:
                        raise ValueError(
                            f"expected one {source_split} video archive member for scene {scene}, "
                            f"found {len(matches)}"
                        )
                    member = matches[0]
                    info = archive.getinfo(member)
                    media = archive.read(info)
                    if len(media) != info.file_size or zlib.crc32(media) & 0xFFFFFFFF != info.CRC:
                        raise ValueError(f"archive CRC/size mismatch for {member}")
                    if len(media) < 16 or b"ftyp" not in media[:16]:
                        raise ValueError(f"invalid MP4 payload for {member}")
                    digest = hashlib.sha256(media).hexdigest()
                    if digest in prior_hashes:
                        raise ValueError(f"selected CLEVRER media was used previously: {member}")
                    split_dir = "train" if source_split == "train" else "validation"
                    media_path = Path("media") / split_dir / str(source_row["video_filename"])
                    target = output_dir / media_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_file():
                        if target.stat().st_size != len(media) or sha256_file(target) != digest:
                            raise ValueError(f"existing partial media hash mismatch: {target}")
                        reused_by_split[source_split] += 1
                    else:
                        target.write_bytes(media)
                        downloaded_by_split[source_split] += 1
                    for partition, rows in selected.items():
                        if source_split == "train" and partition != "train":
                            continue
                        if source_split == "validation" and partition == "train":
                            continue
                        for row in rows:
                            if int(row["scene_index"]) != scene:
                                continue
                            row.update(
                                {
                                    "source": f"MIT-IBM/CLEVRER@{REVISION}",
                                    "scene_group_id": f"clevrer:{source_split}:{scene}",
                                    "media_path": media_path.as_posix(),
                                    "media_bytes": len(media),
                                    "media_sha256": digest,
                                    "archive_member": member,
                                }
                            )
                            media_hashes_by_partition[partition].add(digest)
                    if digest in media_hashes_by_partition["train"]:
                        # The first insertion is valid for the train partition;
                        # cross-partition collisions are checked after all splits.
                        pass
        finally:
            range_bytes_by_split[source_split] = reader.bytes_read
            reader.close()

    media_overlaps = {}
    if holdout_only:
        if previous_sample_dir is None:
            raise ValueError("holdout fetch requires --previous-sample-dir for identity checks")
        previous_hashes = set()
        for manifest_name in ("train.jsonl", "validation.jsonl"):
            previous_hashes.update(
                str(row["media_sha256"])
                for row in _read_jsonl(previous_sample_dir / manifest_name)
            )
        media_overlaps["holdout_previous_train_development"] = sorted(
            media_hashes_by_partition["holdout"] & previous_hashes
        )
    else:
        media_overlaps["train_development"] = sorted(
            media_hashes_by_partition["train"]
            & media_hashes_by_partition["development"]
        )
    if any(media_overlaps.values()):
        raise ValueError("media SHA-256 identity overlaps across fresh partitions")

    # The default mode never reads or fetches the holdout manifest.
    if holdout_only:
        _write_jsonl(output_dir / "holdout.jsonl", selected["holdout"])
    else:
        _write_jsonl(output_dir / "train.jsonl", selected["train"])
        _write_jsonl(output_dir / "validation.jsonl", selected["development"])
    for partition in selected:
        paths = []
        if partition == "train":
            paths = [output_dir / row["media_path"] for row in selected[partition]]
        elif partition == "development":
            paths = [output_dir / row["media_path"] for row in selected[partition]]
        else:
            paths = [output_dir / row["media_path"] for row in selected[partition]]
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                raise FileNotFoundError(f"selected video media is missing: {path}")

    fetch_report = {
        "schema_version": 1,
        "status": "complete_fresh_clevrer_reasoning_media",
        "experiment_id": report_path["experiment_id"],
        "source_revision": REVISION,
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_selection_hashes": {
            name: report_path["partitions"][name]["canonical_rows_sha256"] for name in selected
        },
        "archive_sources": {
            split: {
                "url": url,
                "size_bytes": size,
                "etag": etag,
                "range_bytes_read": range_bytes_by_split[split],
            }
            for split, (url, size, etag) in archives.items()
        },
        "archive_full_sha256_verified": False,
        "downloaded_unique_videos_by_source_split": downloaded_by_split,
        "reused_local_videos_by_source_split": reused_by_split,
        "unique_media_hashes_by_partition": {
            name: len(values) for name, values in media_hashes_by_partition.items()
        },
        "historical_media_hash_count": len(prior_hashes),
        "historical_media_identity_overlap": 0,
        "cross_partition_media_hash_overlap": media_overlaps,
        "scene_overlap": 0,
        "media_hash_overlap": 0,
        "model_selection_manifests": (
            [] if holdout_only else ["train.jsonl", "validation.jsonl"]
        ),
        "holdout_manifest_path": "holdout.jsonl" if holdout_only else None,
        "holdout_fetched_after_selection": holdout_only,
        "holdout_loaded_by_feature_extractor_or_trainer": False,
        "sealed_audit_loaded": False,
    }
    _write_json(output_dir / "fetch-report.json", fetch_report)
    return fetch_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen", type=Path, default=SCREEN_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--holdout-only", action="store_true")
    parser.add_argument("--previous-sample-dir", type=Path)
    args = parser.parse_args()
    report = fetch(
        preflight_dir=args.preflight_dir.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        screen_path=args.screen.resolve(strict=True),
        resume=args.resume,
        holdout_only=args.holdout_only,
        previous_sample_dir=(
            args.previous_sample_dir.resolve(strict=True)
            if args.previous_sample_dir
            else None
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
