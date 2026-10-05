from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.clevrer_quality import (  # noqa: E402
    materialize_clevrer_examples,
    media_path_for_scene,
    select_unseen_clevrer_scenes,
)
from tiny_omni_decision.dataset import iter_local_rows, sha256_file  # noqa: E402
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import DatasetManifest, MediaRef  # noqa: E402

DEFAULT_ARCHIVE_URL = "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip"
DEFAULT_ARCHIVE_SIZE = 12_354_893_389
DEFAULT_QUESTIONS_SHA256 = (
    "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
)
RANGE_BLOCK_BYTES = 2 * 1024 * 1024
MIN_FREE_AFTER_DOWNLOAD = 1024 * 1024 * 1024


class HttpRangeReader(io.RawIOBase):
    """Seekable read-only view of a pinned-size HTTP resource using exact ranges."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0
        self.block_cache: tuple[int, bytes] | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("cannot seek before the start of the archive")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        end = self.position + size
        output = bytearray()
        while self.position < end:
            block_index = self.position // RANGE_BLOCK_BYTES
            block_start = block_index * RANGE_BLOCK_BYTES
            if self.block_cache is None or self.block_cache[0] != block_index:
                block_end = min(block_start + RANGE_BLOCK_BYTES, self.size) - 1
                request = Request(
                    self.url, headers={"Range": f"bytes={block_start}-{block_end}"}
                )
                with urlopen(request, timeout=120) as response:
                    data = response.read()
                    expected_range = f"bytes {block_start}-{block_end}/{self.size}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                        or len(data) != block_end - block_start + 1
                    ):
                        raise OSError("CLEVRER archive returned a non-exact byte range")
                self.block_cache = (block_index, data)
            data = self.block_cache[1]
            offset = self.position - block_start
            take = min(end - self.position, len(data) - offset)
            output.extend(data[offset : offset + take])
            self.position += take
        return bytes(output)


def _scene_ids_from_corpus(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"required historical train/development corpus is missing: {path}")
    scenes: set[int] = set()
    for row in iter_local_rows(path):
        if row.get("modality") == "video" and row.get("source") == "MIT-IBM/CLEVRER":
            source_id = str(row.get("source_record_id", ""))
            try:
                scenes.add(int(source_id.split(":", 1)[0]))
            except ValueError as exc:
                raise ValueError(f"invalid historical CLEVRER source id in {path}") from exc
    return scenes


def _write_examples(path: Path, examples: list[Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.model_dump_json(exclude_none=True) + "\n")


def _verify_existing_archive_member(path: Path, info: zipfile.ZipInfo) -> str:
    if not path.is_file() or path.stat().st_size != info.file_size:
        raise ValueError(f"existing CLEVRER media does not match pinned member size: {path}")
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    if crc & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"existing CLEVRER media failed ZIP CRC validation: {path}")
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    expected_root = (ROOT / "artifacts" / "teacher-quality-next").resolve()
    if not output_dir.is_relative_to(expected_root):
        raise ValueError("output directory must remain under artifacts/teacher-quality-next")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    source_manifest_path = args.source_manifest.resolve()
    manifest = DatasetManifest.model_validate(load_structured_file(source_manifest_path))
    questions_path = args.questions.resolve()
    if sha256_file(questions_path) != DEFAULT_QUESTIONS_SHA256:
        raise ValueError("CLEVRER train questions do not match the pinned source hash")
    rows = list(iter_local_rows(questions_path))

    historical_scenes: set[int] = set()
    for path in args.exclude_corpus:
        historical_scenes.update(_scene_ids_from_corpus(path.resolve()))
    train_scenes, validation_scenes = select_unseen_clevrer_scenes(
        rows,
        previously_observed=historical_scenes,
        train_scene_count=args.train_scenes,
        validation_scene_count=args.validation_scenes,
        seed=args.seed,
    )
    selected_rows = {
        row["scene_index"]: row
        for row in rows
        if row.get("scene_index") in train_scenes | validation_scenes
    }
    archive_url = str(manifest.notes["video_archive_url"])
    archive_size = int(manifest.notes["video_archive_size_bytes"])
    if archive_size != DEFAULT_ARCHIVE_SIZE:
        raise ValueError("pinned CLEVRER archive size differs from the supported source")
    with urlopen(Request(archive_url, method="HEAD"), timeout=30) as response:
        if int(response.headers.get("Content-Length", "-1")) != archive_size:
            raise ValueError("CLEVRER archive length differs from its pinned manifest")
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise ValueError("CLEVRER source does not advertise byte-range support")
    with zipfile.ZipFile(HttpRangeReader(archive_url, archive_size)) as archive:
        by_name = {
            PurePosixPath(item.filename).name: item
            for item in archive.infolist()
            if item.filename.endswith(".mp4")
        }
        selected_infos = {}
        for scene, row in selected_rows.items():
            filename = str(row.get("video_filename", ""))
            info = by_name.get(filename)
            if info is None:
                raise ValueError(f"pinned CLEVRER archive is missing {filename}")
            selected_infos[scene] = info

        media_dir = args.data_root / "raw" / "teacher-quality-next" / "clevrer"
        media_dir.mkdir(parents=True, exist_ok=True)
        media_by_scene: dict[int, MediaRef] = {}
        missing_infos: dict[int, zipfile.ZipInfo] = {}
        for scene, info in selected_infos.items():
            relative_path = media_path_for_scene(scene)
            target = args.data_root / relative_path
            if target.exists():
                digest = _verify_existing_archive_member(target, info)
                media_by_scene[scene] = MediaRef(
                    kind="video",
                    path=relative_path,
                    sha256=digest,
                    license=manifest.license,
                )
            else:
                missing_infos[scene] = info

        expected_bytes = sum(info.file_size for info in missing_infos.values())
        free_bytes = shutil.disk_usage(args.data_root).free
        if free_bytes - expected_bytes < MIN_FREE_AFTER_DOWNLOAD:
            raise OSError(
                "insufficient disk space for selected CLEVRER videos while preserving "
                f"the 1 GiB reserve: need {expected_bytes:,} additional bytes, "
                f"have {free_bytes:,} free"
            )

        print(
            f"Verified {len(media_by_scene)} existing selected videos; "
            f"fetching {len(missing_infos)} more ({expected_bytes:,} bytes).",
            flush=True,
        )
        for scene, info in sorted(missing_infos.items()):
            relative_path = media_path_for_scene(scene)
            target = args.data_root / relative_path
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            written = 0
            try:
                with archive.open(info) as source, temporary.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if written != info.file_size:
                    raise OSError(f"truncated CLEVRER member for scene {scene}")
                temporary.replace(target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            media_by_scene[scene] = MediaRef(
                kind="video",
                path=relative_path,
                sha256=digest.hexdigest(),
                license=manifest.license,
            )

    train, validation, report = materialize_clevrer_examples(
        rows,
        train_scenes=train_scenes,
        validation_scenes=validation_scenes,
        manifest=manifest,
        media_by_scene=media_by_scene,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    _write_examples(train_path, train)
    _write_examples(validation_path, validation)
    result = {
        "schema_version": 1,
        "dataset_id": "teacher-quality-next-clevrer-fresh-scenes-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "source_manifest": source_manifest_path.relative_to(ROOT).as_posix(),
        "source_questions_sha256": sha256_file(questions_path),
        "source_revision": manifest.revision,
        "archive_url": archive_url,
        "archive_size_bytes": archive_size,
        "historical_corpus_exclusions": [path.resolve().as_posix() for path in args.exclude_corpus],
        "historical_scene_count": len(historical_scenes),
        "train_scene_ids": sorted(train_scenes),
        "validation_scene_ids": sorted(validation_scenes),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "train_media_bytes": sum(selected_infos[s].file_size for s in train_scenes),
        "validation_media_bytes": sum(
            selected_infos[s].file_size for s in validation_scenes
        ),
        "media_sha256_by_scene": {
            str(scene): media_by_scene[scene].sha256 for scene in sorted(media_by_scene)
        },
        "task_sampling": "all eligible source rows from selected scenes; no duplication",
        "static_descriptive_questions_included": False,
        "sealed_audit_loaded": False,
        "report": report,
    }
    result["manifest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_dir / "corpus-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build scene-disjoint fresh CLEVRER train/development corpora from official "
            "train data."
        )
    )
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/raw/clevrer/train-questions.json"
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "manifests/candidates/clevrer-video-native.yaml",
    )
    parser.add_argument(
        "--exclude-corpus",
        type=Path,
        action="append",
        default=[
            ROOT / "data/processed/durable-teacher-v1/train.jsonl",
            ROOT / "data/processed/durable-teacher-v1/validation.jsonl",
            ROOT
            / "artifacts/tiny-omni-decision-teacher-v2/candidate-e/corpus-v1-seed17/train.jsonl",
            ROOT
            / "artifacts/tiny-omni-decision-teacher-v2/candidate-e/"
            / "corpus-v1-seed17/validation.jsonl",
        ],
        help="historical non-audit corpus to exclude from fresh scene selection; repeatable",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/teacher-quality-next/clevrer-fresh-scenes-v1",
    )
    parser.add_argument("--train-scenes", type=int, default=500)
    parser.add_argument("--validation-scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
