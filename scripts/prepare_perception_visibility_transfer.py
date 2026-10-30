"""Freeze unused Perception Test TRAIN videos for a fixed-checkpoint transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import requests

from tiny_omni_decision.perception_visibility_transfer import select_unseen_events

SOURCE_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
FIT_COHORT_SHA256 = "dd8c9de6d8970eb1105cb81551e386fca563eb174b8bcda5b09dcc8a16e567be"
TRAIN_ANNOTATION_SHA256 = "0200a50644ab9d36a2dd80e5239e709cabeeaf2e4274eca201c90743b89cc6f5"
ANNOTATION_ROOT = (
    Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
    / "perception-test-mcqa-screen-20261021"
)
FIT_COHORT_PATH = ANNOTATION_ROOT / "fixed_av_cohort_v1_seed17.json"
TRAIN_ANNOTATIONS_PATH = ANNOTATION_ROOT / "sound_train_annotations"
ARCHIVE_BASE = "https://storage.googleapis.com/dm-perception-test/zip_data/"
ARCHIVES = {
    "train_videos.zip": (28_404_087_974, '"-CJXz5tD4nP8CEAE="'),
    "train_audios.zip": (13_222_906_156, '"-CMb2h5uDnf8CEAE="'),
}
EVENTS_PER_CLASS = 128
SELECTION_SEED = 17
CONTEXT_SECONDS = 1.0
MIN_FREE_MARGIN_BYTES = 2 * 1024**3


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PinnedRangeReader(io.RawIOBase):
    """Read a pinned remote ZIP by exact byte ranges without fetching media."""

    def __init__(self, url: str, size: int, etag: str) -> None:
        self.url = url
        self.size = size
        self.etag = etag
        self.position = 0
        self.bytes_read = 0
        self.session = requests.Session()
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        if int(response.headers.get("Content-Length", -1)) != size:
            raise ValueError("pinned archive size mismatch")
        if response.headers.get("ETag") != etag:
            raise ValueError("pinned archive ETag mismatch")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError("invalid seek mode")
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.position >= self.size:
            return b""
        amount = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        end = self.position + amount - 1
        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}",
                "Accept-Encoding": "identity",
                "If-Match": self.etag,
            },
            timeout=90,
        )
        expected_range = f"bytes {self.position}-{end}/{self.size}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected_range
            or response.headers.get("ETag") != self.etag
            or len(response.content) != amount
        ):
            raise RuntimeError("remote ZIP range integrity check failed")
        self.position += amount
        self.bytes_read += amount
        return response.content


def load_inputs(
    reuse_media_root: Path | None,
    reuse_acquisition_path: Path | None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    annotation_sha = sha256_file(TRAIN_ANNOTATIONS_PATH)
    if annotation_sha != TRAIN_ANNOTATION_SHA256:
        raise ValueError("pinned train-annotation archive hash mismatch")
    fit_bytes = FIT_COHORT_PATH.read_bytes()
    fit_sha = sha256_bytes(fit_bytes)
    if fit_sha != FIT_COHORT_SHA256:
        raise ValueError("original fit cohort hash mismatch")
    fit = json.loads(fit_bytes)
    fit_rows = fit["splits"]["train"]
    if len(fit_rows) != 256 or sum(row["target"] == 0 for row in fit_rows) != 128:
        raise ValueError("unexpected original fit cohort size or class balance")
    if sum(row["target"] == 1 for row in fit_rows) != 128:
        raise ValueError("unexpected original fit cohort size or class balance")
    with ZipFile(TRAIN_ANNOTATIONS_PATH) as archive:
        members = archive.namelist()
        if members != ["sound_localisation_train.json"]:
            raise ValueError("unexpected train annotation archive members")
        annotation_bytes = archive.read(members[0])
    annotations = json.loads(annotation_bytes)
    selected = select_unseen_events(
        annotations,
        excluded_video_ids={row["video_id"] for row in fit_rows},
        source_split="train",
        seed=SELECTION_SEED,
        events_per_class=EVENTS_PER_CLASS,
        context_seconds=CONTEXT_SECONDS,
    )
    if len(selected) != 2 * EVENTS_PER_CLASS:
        raise ValueError("transfer cohort size mismatch")
    if set(row["video_id"] for row in selected) & {row["video_id"] for row in fit_rows}:
        raise ValueError("transfer cohort overlaps the readout fit videos")
    if {row["target"] for row in selected} != {0, 1}:
        raise ValueError("transfer cohort is missing a class")
    cohort = {
        "schema_version": 1,
        "source": "Google DeepMind Perception Test",
        "source_revision": SOURCE_REVISION,
        "source_split": "train",
        "diagnostic_scope": (
            "video-disjoint transfer within official TRAIN only; "
            "not blind or independent validation"
        ),
        "label_read_status": (
            "training annotations read to make this predeclared deterministic selection"
        ),
        "selection_seed": SELECTION_SEED,
        "events_per_class": EVENTS_PER_CLASS,
        "boundary_context_seconds": CONTEXT_SECONDS,
        "selection_rule": (
            "select one SHA-256-ranked eligible event per video before stratifying; "
            "exclude all original fit video IDs; rank remaining groups by a "
            "domain-separated SHA-256 key; take exactly 128 per label"
        ),
        "original_fit_cohort_sha256": fit_sha,
        "source_train_annotation_archive_sha256": annotation_sha,
        "source_train_annotation_json_sha256": sha256_bytes(annotation_bytes),
        "fit_video_overlap": 0,
        "examples": selected,
    }
    reuse_receipt = None
    if (reuse_media_root is None) != (reuse_acquisition_path is None):
        raise ValueError("reuse media root and acquisition receipt must be supplied together")
    if reuse_acquisition_path is not None:
        reuse_receipt = json.loads(reuse_acquisition_path.read_text(encoding="utf-8"))
        if reuse_receipt.get("status") != "PASS_TRAIN_MEDIA_ACQUISITION":
            raise ValueError("existing media acquisition receipt is not complete")
    return cohort, selected, {
        "fit_video_ids": {row["video_id"] for row in fit_rows},
        "train_annotation_json_bytes": len(annotation_bytes),
        "reuse_media_root": reuse_media_root,
        "reuse_receipt": reuse_receipt,
    }


def inspect_remote_archives(selected: list[dict[str, object]]) -> dict[str, object]:
    results: dict[str, object] = {}
    for archive_name, (size, etag) in ARCHIVES.items():
        reader = PinnedRangeReader(ARCHIVE_BASE + archive_name, size, etag)
        modality = "videos" if archive_name.endswith("videos.zip") else "audios"
        extension = "mp4" if modality == "videos" else "wav"
        with ZipFile(reader) as archive:
            members = []
            for row in selected:
                member = f"{modality}/{row['video_id']}.{extension}"
                info = archive.getinfo(member)
                members.append(
                    {
                        "video_id": row["video_id"],
                        "member": member,
                        "compressed_bytes": info.compress_size,
                        "uncompressed_bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                        "header_offset": info.header_offset,
                    }
                )
        results[archive_name] = {
            "size": size,
            "etag": etag,
            "files": len(members),
            "directory_bytes_read": reader.bytes_read,
            "compressed_bytes": sum(row["compressed_bytes"] for row in members),
            "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in members),
            "members": members,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02"),
    )
    parser.add_argument("--reuse-media-root", type=Path)
    parser.add_argument("--reuse-acquisition", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite transfer attempt: {args.output}")
    cohort, selected, inputs = load_inputs(args.reuse_media_root, args.reuse_acquisition)
    archive_preflight = inspect_remote_archives(selected)
    prior_files = {
        (row["video_id"], Path(row["member"]).suffix): row
        for row in (inputs["reuse_receipt"] or {}).get("files", [])
    }
    reuse_root = inputs["reuse_media_root"]
    missing_ids: set[str] = set()
    reused_ids: set[str] = set()
    expected_member_bytes: dict[tuple[str, str], tuple[int, str]] = {}
    for archive_data in archive_preflight.values():
        for row in archive_data["members"]:
            extension = Path(row["member"]).suffix
            expected_member_bytes[(row["video_id"], extension)] = (
                row["uncompressed_bytes"],
                row["crc32"],
            )
    for video_id in {row["video_id"] for row in selected}:
        existing = []
        for extension in (".mp4", ".wav"):
            path = reuse_root / "train" / f"{video_id}{extension}" if reuse_root else None
            prior = prior_files.get((video_id, extension))
            if path is not None and path.is_file() and prior is not None:
                expected_size, _ = expected_member_bytes[(video_id, extension)]
                if path.stat().st_size != expected_size or sha256_file(path) != prior.get("sha256"):
                    raise ValueError(f"existing media hash or size mismatch: {path}")
                existing.append(True)
            else:
                existing.append(False)
        if all(existing):
            reused_ids.add(video_id)
        elif any(existing):
            raise ValueError(f"only one media modality is available for existing video {video_id}")
        else:
            missing_ids.add(video_id)
    expanded_bytes = sum(
        row["uncompressed_bytes"]
        for archive_data in archive_preflight.values()
        for row in archive_data["members"]
        if row["video_id"] in missing_ids
    )
    disk = shutil.disk_usage(args.output.parent)
    required_bytes = expanded_bytes + MIN_FREE_MARGIN_BYTES
    if disk.free < required_bytes:
        raise OSError(
            "insufficient disk for additional transfer media: "
            f"free={disk.free}, need={required_bytes}"
        )
    cohort["archive_preflight"] = archive_preflight
    cohort["archive_preflight_total_uncompressed_bytes"] = sum(
        result["uncompressed_bytes"] for result in archive_preflight.values()
    )
    cohort["archive_preflight_additional_uncompressed_bytes"] = expanded_bytes
    cohort["reused_media_video_count"] = len(reused_ids)
    cohort["new_media_video_count"] = len(missing_ids)
    cohort["preflight_disk_free_bytes"] = disk.free
    cohort["minimum_free_margin_bytes"] = MIN_FREE_MARGIN_BYTES
    cohort["media_payload_downloaded"] = False
    cohort["training_or_inference_performed"] = False
    cohort_bytes = (json.dumps(cohort, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "transfer_cohort.json").write_bytes(cohort_bytes)
    report = {
        "status": "PASS_TRANSFER_COHORT_FROZEN_ARCHIVE_PREFLIGHT",
        "source_revision": SOURCE_REVISION,
        "transfer_cohort_sha256": sha256_bytes(cohort_bytes),
        "original_fit_video_groups": len(inputs["fit_video_ids"]),
        "selected_transfer_groups": len(selected),
        "selected_class_counts": {
            "not_visible": sum(row["target"] == 0 for row in selected),
            "visible": sum(row["target"] == 1 for row in selected),
        },
        "fit_video_overlap": 0,
        "source_split": "official train only",
        "archive_preflight": {
            name: {key: value for key, value in data.items() if key != "members"}
            for name, data in archive_preflight.items()
        },
        "expected_additional_media_bytes": expanded_bytes,
        "reused_media_video_count": len(reused_ids),
        "new_media_video_count": len(missing_ids),
        "disk_free_at_preflight_bytes": disk.free,
        "training_or_model_evaluation_performed": False,
    }
    (args.output / "preflight.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
