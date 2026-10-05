from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.dataset import iter_local_rows, sha256_file  # noqa: E402
from tiny_omni_decision.librispeech import (  # noqa: E402
    build_librispeech_development_decisions,
    normalize_transcript,
)

DATA_ROOT = ROOT / "data"
RAW_DIR = DATA_ROOT / "raw/teacher-quality-next/librispeech-dev-other"
OUTPUT_DIR = DATA_ROOT / "processed/teacher-quality-next/librispeech-dev-other"
BASE_URL = "https://www.openslr.org/resources/12"
ARCHIVE_NAME = "dev-other.tar.gz"
EXPECTED_MD5 = "c8d0bcc9cca99d4f8b62fcc847357931"
EXPECTED_MANIFEST_SHA256 = "efe6522682d078fdcfd2c3b2d418a53a89683b342dd5147a769a41a7e021fe3f"
TRAIN_CORPUS = DATA_ROOT / "processed/teacher-quality-next/multimodal-data-coverage-v1/train.jsonl"


def _fetch_pinned_archive() -> tuple[Path, str, str, str]:
    checksum_url = f"{BASE_URL}/md5sum.txt"
    with urlopen(checksum_url, timeout=30) as response:
        checksum_manifest = response.read()
    manifest_sha256 = hashlib.sha256(checksum_manifest).hexdigest()
    manifest_revision = hashlib.sha1(checksum_manifest).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("official SLR12 checksum manifest differs from the pinned snapshot")
    entries = {
        filename: digest
        for digest, filename in (
            line.split(maxsplit=1)
            for line in checksum_manifest.decode("ascii").splitlines()
        )
    }
    if entries.get(ARCHIVE_NAME) != EXPECTED_MD5:
        raise ValueError("official SLR12 checksum manifest does not match pinned dev-other MD5")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / ARCHIVE_NAME
    if archive_path.exists():
        if _md5_file(archive_path) != EXPECTED_MD5:
            raise ValueError("existing dev-other archive fails the official MD5")
        return archive_path, sha256_file(archive_path), manifest_sha256, manifest_revision
    temporary = archive_path.with_suffix(archive_path.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"incomplete archive remains; inspect before retrying: {temporary}")
    digest = hashlib.md5()
    sha256 = hashlib.sha256()
    request = Request(
        f"{BASE_URL}/{ARCHIVE_NAME}", headers={"User-Agent": "tiny-omni-decision/1.0"}
    )
    with urlopen(request, timeout=90) as response, temporary.open("xb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        written = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            sha256.update(chunk)
            written += len(chunk)
            if written % (32 * 1024 * 1024) < len(chunk) or written == total:
                print(f"Downloaded dev-other: {written}/{total} bytes", flush=True)
    if digest.hexdigest() != EXPECTED_MD5:
        temporary.unlink(missing_ok=True)
        raise ValueError("downloaded dev-other archive fails the official MD5")
    temporary.replace(archive_path)
    return archive_path, sha256.hexdigest(), manifest_sha256, manifest_revision


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_transcriptions(archive_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".trans.txt"):
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) != 5 or parts[:2] != ("LibriSpeech", "dev-other"):
                raise ValueError(f"unexpected dev-other transcript path: {member.name}")
            speaker_id, chapter_id = parts[2], parts[3]
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"could not read transcript file {member.name}")
            for raw in source:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                utterance_id, transcript = line.split(" ", 1)
                if not utterance_id.startswith(f"{speaker_id}-{chapter_id}-"):
                    raise ValueError(f"utterance ID does not match chapter: {utterance_id}")
                media_path = (
                    RAW_DIR.relative_to(DATA_ROOT).as_posix()
                    + f"/audio/dev-other/{speaker_id}/{chapter_id}/{utterance_id}.flac"
                )
                records.append(
                    {
                        "split": "validation",
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utterance_id,
                        "transcript": transcript,
                        "archive_member": "/".join((*parts[:-1], f"{utterance_id}.flac")),
                        "media_path": media_path,
                    }
                )
    return records


def _extract_audio(archive_path: Path, records: list[dict[str, str]]) -> None:
    wanted = {row["archive_member"]: row for row in records}
    found: set[str] = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            row = wanted.get(member.name)
            if row is None:
                continue
            if not member.isfile() or not member.name.endswith(".flac"):
                raise ValueError(f"expected a FLAC audio member: {member.name}")
            target = DATA_ROOT / row["media_path"]
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"could not extract audio member {member.name}")
            digest = hashlib.sha256()
            if target.exists():
                if not target.is_file() or target.is_symlink():
                    raise ValueError(f"unexpected existing dev-other media path: {target}")
                with source, target.open("rb") as existing:
                    while chunk := existing.read(1024 * 1024):
                        if source.read(len(chunk)) != chunk:
                            raise ValueError(f"existing dev-other media differs: {target}")
                        digest.update(chunk)
                    if source.read(1):
                        raise ValueError(f"existing dev-other media is truncated: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".flac.part")
                with temporary.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                temporary.replace(target)
            row["media_sha256"] = digest.hexdigest()
            found.add(member.name)
    if found != wanted.keys():
        raise ValueError(f"missing dev-other audio members: {len(wanted.keys() - found)}")


def build() -> dict[str, object]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus: {OUTPUT_DIR}")
    archive_path, archive_sha256, manifest_sha256, revision = _fetch_pinned_archive()
    records = _read_transcriptions(archive_path)
    if not records:
        raise ValueError("official dev-other archive has no transcript records")

    train_rows = [
        row
        for row in iter_local_rows(TRAIN_CORPUS)
        if row.get("source") == "openslr/LibriSpeech"
    ]
    train_transcripts = {
        normalize_transcript(str(row.get("source_target", row.get("target", ""))))
        for row in train_rows
    }
    validation_transcripts = {normalize_transcript(row["transcript"]) for row in records}
    transcript_overlap = train_transcripts & validation_transcripts
    if transcript_overlap:
        raise ValueError(
            f"dev-other overlaps existing LibriSpeech training content: {len(transcript_overlap)}"
        )
    train_speakers = {
        str(row["task_group_id"]).removeprefix("speaker:") for row in train_rows
    }
    validation_speakers = {row["speaker_id"] for row in records}
    speaker_overlap = train_speakers & validation_speakers
    if speaker_overlap:
        raise ValueError(
            "dev-other overlaps existing LibriSpeech train speakers: "
            f"{len(speaker_overlap)}"
        )
    train_record_ids = {str(row.get("source_record_id", "")) for row in train_rows}
    record_overlap = train_record_ids & {row["utterance_id"] for row in records}
    if record_overlap:
        raise ValueError(f"dev-other overlaps existing LibriSpeech records: {len(record_overlap)}")

    _extract_audio(archive_path, records)
    train_audio_hashes = {
        str(media.get("sha256"))
        for row in train_rows
        for media in row.get("media", [])
        if media.get("kind") == "audio" and media.get("sha256")
    }
    validation_audio_hashes = {row["media_sha256"] for row in records}
    media_overlap = train_audio_hashes & validation_audio_hashes
    if media_overlap:
        raise ValueError(
            "dev-other overlaps existing LibriSpeech audio assets: "
            f"{len(media_overlap)}"
        )
    examples, accounting = build_librispeech_development_decisions(
        records, revision=revision, seed=29
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    corpus_path = OUTPUT_DIR / "validation.jsonl"
    with corpus_path.open("x", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.model_dump_json(exclude_none=True) + "\n")
    result: dict[str, object] = {
        "dataset": "OpenSLR SLR12 LibriSpeech dev-other",
        "license": "CC-BY-4.0",
        "source_revision": revision,
        "checksum_manifest_sha256": manifest_sha256,
        "official_archive_md5": EXPECTED_MD5,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "seed": 29,
        "accounting": accounting,
        "normalized_transcript_overlap_with_existing_librispeech_train": 0,
        "speaker_overlap_with_existing_librispeech_train": 0,
        "source_record_overlap_with_existing_librispeech_train": 0,
        "media_sha256_overlap_with_existing_librispeech_train": 0,
        "validation_sha256": sha256_file(corpus_path),
        "sealed_audit_loaded": False,
        "test_archives_downloaded_or_read": False,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
