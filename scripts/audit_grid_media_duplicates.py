"""Audit exact audio/video payload duplicates across the corrected GRID split."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath

from scripts.acquire_grid_video_archives import file_hashes

REPO = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024


def digest_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def archive_member_hashes(path: Path, suffix: str) -> dict[str, str]:
    result = {}
    if suffix == ".wav":
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                if not member.isfile() or member_path.suffix.lower() != suffix:
                    continue
                code = member_path.stem
                if code in result:
                    raise ValueError(f"duplicate {suffix} payload for {code} in {path}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"missing payload stream for {member.name}")
                with stream:
                    result[code] = digest_stream(stream)
    elif suffix == ".mpg":
        with zipfile.ZipFile(path) as archive:
            seen_names = set()
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if member.is_dir() or member_path.suffix.lower() != suffix:
                    continue
                if member_path.name.startswith("._") or "__MACOSX" in member_path.parts:
                    continue
                if member.filename in seen_names:
                    raise ValueError(f"duplicate archive member name: {member.filename}")
                seen_names.add(member.filename)
                code = member_path.stem
                if code in result:
                    raise ValueError(f"duplicate {suffix} payload for {code} in {path}")
                with archive.open(member) as stream:
                    result[code] = digest_stream(stream)
    else:
        raise ValueError(f"unsupported media suffix: {suffix}")
    return result


def find_duplicates(records: list[dict]) -> dict:
    by_hash: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_hash.setdefault((record["modality"], record["sha256"]), []).append(record)
    cross_split = []
    within_split = []
    for (modality, sha256), matches in sorted(by_hash.items()):
        if len(matches) < 2:
            continue
        splits = {match["split"] for match in matches}
        finding = {
            "modality": modality,
            "sha256": sha256,
            "occurrences": matches,
        }
        if len(splits) > 1:
            cross_split.append(finding)
        else:
            within_split.append(finding)
    return {
        "cross_split_exact_payload_duplicates": cross_split,
        "within_split_exact_payload_duplicates": within_split,
    }


def _load_rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"split manifest must be a JSON list: {path}")
    for row in raw:
        if not {"speaker", "code", "targets"} <= row.keys():
            raise ValueError(f"invalid row in {path}")
    return raw


def run_audio_audit(
    *,
    summary_path: Path,
    expected_summary_sha256: str,
    bundle_index_path: Path,
    train_path: Path,
    validation_path: Path,
    output_path: Path,
) -> dict:
    """Finish the audio duplicate gate while video ZIP acquisition continues."""
    output_path = output_path.resolve()
    if output_path.is_relative_to(REPO) or output_path.exists():
        raise ValueError("use a new output file outside the repository")
    if file_hashes(summary_path)[1] != expected_summary_sha256:
        raise ValueError("source summary SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if file_hashes(bundle_index_path)[1] != summary.get("bundle_index_sha256"):
        raise ValueError("bundle index SHA-256 differs from split receipt")
    if file_hashes(train_path)[1] != summary.get("output_sha256", {}).get("train"):
        raise ValueError("train manifest SHA-256 differs from split receipt")
    if file_hashes(validation_path)[1] != summary.get("output_sha256", {}).get("validation"):
        raise ValueError("validation manifest SHA-256 differs from split receipt")
    bundle_index = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    train_rows, validation_rows = _load_rows(train_path), _load_rows(validation_path)
    if {row["code"] for row in train_rows} & {row["code"] for row in validation_rows}:
        raise ValueError("train/validation source code overlap")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        if {row["speaker"] for row in rows} != set(
            summary["frozen_speaker_assignment"][split]
        ):
            raise ValueError(f"{split} speaker assignment differs from frozen summary")
    records, per_speaker = [], {}
    started = time.monotonic()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        rows_by_speaker: dict[str, set[str]] = {}
        for row in rows:
            rows_by_speaker.setdefault(row["speaker"], set()).add(row["code"])
        for speaker, expected_codes in sorted(rows_by_speaker.items()):
            bundle = bundle_index["speakers"][speaker]
            audio_path = Path(bundle["audio_tar"]).resolve()
            if file_hashes(audio_path)[1] != bundle["audio_sha256"]:
                raise ValueError(f"audio TAR SHA-256 mismatch: {speaker}")
            audio_hashes = archive_member_hashes(audio_path, ".wav")
            expected_source_assets = int(
                summary["source_receipts"][speaker]["codes_in_audio_alignment_video"]
            )
            if len(audio_hashes) != expected_source_assets:
                raise ValueError(f"incomplete audio payload inventory for {speaker}")
            if not expected_codes <= audio_hashes.keys():
                raise ValueError(f"audio TAR lacks split rows for {speaker}")
            for code in sorted(expected_codes):
                records.append({
                    "split": split,
                    "speaker": speaker,
                    "code": code,
                    "modality": "audio",
                    "sha256": audio_hashes[code],
                })
            per_speaker[speaker] = {
                "split": split,
                "manifest_rows": len(expected_codes),
                "audio_tar_sha256": bundle["audio_sha256"],
                "audio_payloads_hashed": len(audio_hashes),
            }
    duplicates = find_duplicates(records)
    report = {
        "status": (
            "PASS_NO_CROSS_SPLIT_EXACT_AUDIO_DUPLICATES"
            if not duplicates["cross_split_exact_payload_duplicates"]
            else "FAIL_CROSS_SPLIT_EXACT_AUDIO_DUPLICATES"
        ),
        "scope": "exact SHA-256 of WAV member payloads as stored; not re-encoding invariant",
        "source_summary_sha256": expected_summary_sha256,
        "bundle_index_sha256": file_hashes(bundle_index_path)[1],
        "train_manifest_sha256": file_hashes(train_path)[1],
        "validation_manifest_sha256": file_hashes(validation_path)[1],
        "audio_payload_count": len(records),
        "cross_split_exact_payload_duplicate_groups": len(
            duplicates["cross_split_exact_payload_duplicates"]
        ),
        "within_split_exact_payload_duplicate_groups": len(
            duplicates["within_split_exact_payload_duplicates"]
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "per_speaker": per_speaker,
        **duplicates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_audit(
    *,
    summary_path: Path,
    expected_summary_sha256: str,
    bundle_index_path: Path,
    archive_result_path: Path,
    train_path: Path,
    validation_path: Path,
    output_path: Path,
) -> dict:
    output_path = output_path.resolve()
    if output_path.is_relative_to(REPO) or output_path.exists():
        raise ValueError("use a new output file outside the repository")
    summary_md5, summary_sha256 = file_hashes(summary_path)
    del summary_md5
    if summary_sha256 != expected_summary_sha256:
        raise ValueError("source summary SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if file_hashes(bundle_index_path)[1] != summary.get("bundle_index_sha256"):
        raise ValueError("bundle index SHA-256 differs from split receipt")
    if file_hashes(train_path)[1] != summary.get("output_sha256", {}).get("train"):
        raise ValueError("train manifest SHA-256 differs from split receipt")
    if file_hashes(validation_path)[1] != summary.get("output_sha256", {}).get("validation"):
        raise ValueError("validation manifest SHA-256 differs from split receipt")
    bundle_index = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    acquired = json.loads(archive_result_path.read_text(encoding="utf-8"))
    if acquired.get("status") != "PASS_ARCHIVES_VERIFIED":
        raise ValueError("video acquisition result is not fully verified")
    if acquired.get("source_summary_sha256") != summary_sha256:
        raise ValueError("video archives reference a different split summary")
    if len(acquired.get("archives", [])) != 32:
        raise ValueError("expected 32 verified video archives")
    train_rows = _load_rows(train_path)
    validation_rows = _load_rows(validation_path)
    actual_train = {row["code"] for row in train_rows}
    actual_validation = {row["code"] for row in validation_rows}
    if actual_train & actual_validation:
        raise ValueError("train/validation source code overlap")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        expected_speakers = set(summary["frozen_speaker_assignment"][split])
        if {row["speaker"] for row in rows} != expected_speakers:
            raise ValueError(f"{split} speaker assignment differs from frozen summary")

    acquired_by_speaker = {row["speaker"]: row for row in acquired["archives"]}
    if set(acquired_by_speaker) != set(summary["source_receipts"]):
        raise ValueError("verified archive speakers differ from frozen source receipt")
    source_bundles = bundle_index["speakers"]
    records = []
    per_speaker = {}
    started = time.monotonic()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        rows_by_speaker: dict[str, set[str]] = {}
        for row in rows:
            rows_by_speaker.setdefault(row["speaker"], set()).add(row["code"])
        for speaker in sorted(rows_by_speaker):
            expected_codes = rows_by_speaker[speaker]
            bundle = source_bundles[speaker]
            video = acquired_by_speaker[speaker]
            video_path = Path(video["path"]).resolve()
            audio_path = Path(bundle["audio_tar"]).resolve()
            source_video = summary["source_receipts"][speaker]
            if (
                video_path.stat().st_size != source_video["video_archive_bytes"]
                or video.get("publisher_md5")
                != source_video["video_publisher_md5"].removeprefix("md5:")
            ):
                raise ValueError(f"video archive receipt mismatch: {speaker}")
            video_md5, video_sha256 = file_hashes(video_path)
            if video_sha256 != video["sha256"]:
                raise ValueError(f"video archive SHA-256 changed after acquisition: {speaker}")
            if video_md5 != source_video["video_publisher_md5"].removeprefix("md5:"):
                raise ValueError(f"video archive publisher MD5 mismatch: {speaker}")
            if file_hashes(audio_path)[1] != bundle["audio_sha256"]:
                raise ValueError(f"audio TAR SHA-256 mismatch: {speaker}")
            video_hashes = archive_member_hashes(video_path, ".mpg")
            audio_hashes = archive_member_hashes(audio_path, ".wav")
            expected_source_assets = int(
                summary["source_receipts"][speaker]["codes_in_audio_alignment_video"]
            )
            if (
                len(video_hashes) != expected_source_assets
                or len(audio_hashes) != expected_source_assets
            ):
                raise ValueError(f"incomplete media payload inventory for {speaker}")
            if not expected_codes <= video_hashes.keys():
                raise ValueError(f"video archive lacks split rows for {speaker}")
            if not expected_codes <= audio_hashes.keys():
                raise ValueError(f"audio TAR lacks split rows for {speaker}")
            # Only eligible split rows are compared; excluded source codes are
            # not silently added back to either split.
            for code in sorted(expected_codes):
                records.append({
                    "split": split,
                    "speaker": speaker,
                    "code": code,
                    "modality": "audio",
                    "sha256": audio_hashes[code],
                })
                records.append({
                    "split": split,
                    "speaker": speaker,
                    "code": code,
                    "modality": "video",
                    "sha256": video_hashes[code],
                })
            per_speaker[speaker] = {
                "split": split,
                "manifest_rows": len(expected_codes),
                "audio_tar_sha256": bundle["audio_sha256"],
                "video_archive_sha256": video["sha256"],
                "audio_payloads_hashed": len(audio_hashes),
                "video_payloads_hashed": len(video_hashes),
            }
    duplicate_result = find_duplicates(records)
    report = {
        "status": (
            "PASS_NO_CROSS_SPLIT_EXACT_MEDIA_DUPLICATES"
            if not duplicate_result["cross_split_exact_payload_duplicates"]
            else "FAIL_CROSS_SPLIT_EXACT_MEDIA_DUPLICATES"
        ),
        "scope": (
            "exact SHA-256 of WAV/MPG member payloads as stored; "
            "not perceptual or re-encoding invariant"
        ),
        "source_summary_sha256": summary_sha256,
        "bundle_index_sha256": file_hashes(bundle_index_path)[1],
        "archive_result_sha256": file_hashes(archive_result_path)[1],
        "train_manifest_sha256": file_hashes(train_path)[1],
        "validation_manifest_sha256": file_hashes(validation_path)[1],
        "asset_payload_count": len(records),
        "audio_payload_count": sum(record["modality"] == "audio" for record in records),
        "video_payload_count": sum(record["modality"] == "video" for record in records),
        "cross_split_exact_payload_duplicate_groups": len(
            duplicate_result["cross_split_exact_payload_duplicates"]
        ),
        "within_split_exact_payload_duplicate_groups": len(
            duplicate_result["within_split_exact_payload_duplicates"]
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "per_speaker": per_speaker,
        **duplicate_result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--bundle-index", type=Path, required=True)
    parser.add_argument("--archive-result", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()
    if not args.audio_only and args.archive_result is None:
        parser.error("--archive-result is required unless --audio-only is set")
    runner = run_audio_audit if args.audio_only else run_audit
    kwargs = {
        "summary_path": args.summary,
        "expected_summary_sha256": args.expected_summary_sha256,
        "bundle_index_path": args.bundle_index,
        "train_path": args.train,
        "validation_path": args.validation,
        "output_path": args.output,
    }
    if not args.audio_only:
        kwargs["archive_result_path"] = args.archive_result
    report = runner(
        **kwargs,
    )
    print(json.dumps({key: report[key] for key in (
        "status",
        "asset_payload_count" if not args.audio_only else "audio_payload_count",
        "cross_split_exact_payload_duplicate_groups",
        "within_split_exact_payload_duplicate_groups",
        "elapsed_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()
