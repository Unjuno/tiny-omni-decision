"""Build a new GRID development split from verified per-speaker source bundles."""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from pathlib import Path, PurePosixPath

from scripts.audit_droid_train_temporal import digest
from scripts.audit_grid_alignment_content import words_match
from scripts.audit_grid_direct_speaker_codes import tar_codes, zip_codes_from_tail
from scripts.lombard_grid_corpus import decode_code
from scripts.prepare_grid_development_split import PINS, allocate

REPO = Path(__file__).resolve().parents[1]
TARGET_ORDER = ("command", "color", "preposition", "letter", "number", "adverb")
ZENODO_RECORD_SHA256 = "fbcf06eecb05bdb8cb93a4b1cd6d1d4bfad81f5a1545e008c137d23bf1265c7e"


def normalize_targets(targets: dict[str, str]) -> str:
    if set(targets) != set(TARGET_ORDER):
        raise ValueError("targets must contain the six GRID decision slots")
    return " ".join(str(targets[key]).strip().lower() for key in TARGET_ORDER)


def build_split_rows(
    speaker_codes: dict[str, set[str]],
    split_speakers: dict[str, list[str]],
    *,
    forbidden_codes: set[str],
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    if set(split_speakers) != {"train", "validation"}:
        raise ValueError("exact train and validation speaker partitions are required")
    train_speakers, validation_speakers = map(
        set, (split_speakers["train"], split_speakers["validation"])
    )
    if train_speakers & validation_speakers:
        raise ValueError("speaker overlap between train and validation")
    if train_speakers | validation_speakers != set(speaker_codes):
        raise ValueError("speaker inventory and split partitions differ")
    if any(len(values) != len(split_speakers[name])
           for name, values in (("train", train_speakers),
                                ("validation", validation_speakers))):
        raise ValueError("duplicate speaker in a split partition")

    rows: dict[str, list[dict]] = {"train": [], "validation": []}
    excluded = collections.Counter()
    seen_codes: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    for split in ("train", "validation"):
        for speaker in sorted(split_speakers[split]):
            for code in sorted(speaker_codes[speaker]):
                if code in seen_codes:
                    raise ValueError(
                        f"duplicate source code {code}: {seen_codes[code]} and {speaker}"
                    )
                targets = decode_code(code)
                normalized = normalize_targets(targets)
                if normalized in seen_content:
                    raise ValueError(
                        f"duplicate normalized content: {seen_content[normalized]} and {code}"
                    )
                seen_codes[code] = speaker
                seen_content[normalized] = code
                if code in forbidden_codes:
                    excluded[split] += 1
                    continue
                rows[split].append({"speaker": speaker, "code": code, "targets": targets})

    train_codes = {row["code"] for row in rows["train"]}
    validation_codes = {row["code"] for row in rows["validation"]}
    if train_codes & validation_codes:
        raise ValueError("train/validation normalized content overlap")
    return rows, {split: excluded[split] for split in ("train", "validation")}


def _read_alignment_codes(path: Path) -> tuple[set[str], int]:
    codes = set()
    checked = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or PurePosixPath(member.name).suffix.lower() != ".align":
                continue
            if member.size > 4096:
                raise ValueError(f"oversized alignment: {member.name}")
            code = PurePosixPath(member.name).stem
            if code in codes:
                raise ValueError(f"duplicate alignment code for one speaker: {code}")
            targets = decode_code(code)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"alignment payload missing: {member.name}")
            lines = stream.read().decode("utf-8").splitlines()
            words = [
                line.split()[-1] for line in lines
                if line.split() and line.split()[-1] not in {"sil", "sp"}
            ]
            if not words_match(words, targets):
                raise ValueError(f"alignment and code disagree: {member.name}")
            codes.add(code)
            checked += 1
    return codes, checked


def _validate_bundle_index(
    index: dict, zenodo_record: dict, speakers: list[str], *, zenodo_sha256: str,
) -> dict:
    if index.get("source_record") != "doi:10.5281/zenodo.3625687":
        raise ValueError("unexpected source record")
    if index.get("zenodo_record_sha256") != zenodo_sha256:
        raise ValueError("bundle index references different Zenodo metadata")
    bundles = index.get("speakers")
    if not isinstance(bundles, dict) or set(bundles) != set(speakers):
        raise ValueError("bundle index must cover exactly the frozen 32 speakers")
    files = {item["key"]: item for item in zenodo_record.get("files", [])}
    if not files:
        raise ValueError("pinned Zenodo metadata contains no file records")
    return files


def prepare_direct_source_split(
    *,
    bundle_index_path: Path,
    zenodo_record_path: Path,
    frozen_summary_path: Path,
    frozen_train_path: Path,
    frozen_validation_path: Path,
    output_dir: Path,
) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_relative_to(REPO):
        raise ValueError("use a new external output directory")

    bundle_index_sha = digest(bundle_index_path)
    zenodo_sha = digest(zenodo_record_path)
    frozen_summary_sha = digest(frozen_summary_path)
    if zenodo_sha != ZENODO_RECORD_SHA256:
        raise ValueError("pinned Zenodo metadata hash mismatch")
    zenodo_record = json.loads(zenodo_record_path.read_bytes())
    frozen_summary = json.loads(frozen_summary_path.read_bytes())
    frozen_train_sha = digest(frozen_train_path)
    frozen_validation_sha = digest(frozen_validation_path)
    if frozen_train_sha != frozen_summary["output_hashes"]["train"]:
        raise ValueError("frozen train manifest differs from its receipt")
    if frozen_validation_sha != frozen_summary["output_hashes"]["validation"]:
        raise ValueError("frozen validation manifest differs from its receipt")

    expected_pins = {str(path): sha for path, sha in PINS.items()}
    if frozen_summary.get("input_hashes") != expected_pins:
        raise ValueError("frozen split exclusion inputs differ from their receipt")
    for path, expected_sha in PINS.items():
        if not path.exists() or digest(path) != expected_sha:
            raise ValueError(f"pinned source changed or is missing: {path}")

    split_speakers = frozen_summary["speakers"]
    speakers = sorted(set(split_speakers["train"]) | set(split_speakers["validation"]))
    if split_speakers != allocate(speakers):
        raise ValueError("frozen speaker assignment differs from deterministic protocol")
    bundle_index = json.loads(bundle_index_path.read_bytes())
    zenodo_files = _validate_bundle_index(
        bundle_index,
        zenodo_record,
        speakers,
        zenodo_sha256=digest(zenodo_record_path),
    )

    speaker_codes = {}
    source_receipts = {}
    alignment_counts = {}
    source_paths = {}
    for speaker in speakers:
        bundle = bundle_index["speakers"][speaker]
        expected_urls = {
            "audio_url": f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker}/audio/{speaker}.tar",
            "alignment_url": f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker}/align/{speaker}.tar",
            "video_url": f"https://zenodo.org/api/records/3625687/files/{speaker}.zip/content",
        }
        if any(bundle.get(key) != value for key, value in expected_urls.items()):
            raise ValueError(f"unexpected official source URL for {speaker}")
        paths = {key: Path(bundle[key]).resolve() for key in (
            "audio_tar", "alignment_tar", "video_metadata_tail",
        )}
        expected_hashes = {
            "audio_tar": bundle["audio_sha256"],
            "alignment_tar": bundle["alignment_sha256"],
            "video_metadata_tail": bundle["video_metadata_sha256"],
        }
        for key, path in paths.items():
            if not path.is_file() or digest(path) != expected_hashes[key]:
                raise ValueError(f"source artifact missing or hash mismatch: {speaker}/{key}")
        source_paths[speaker] = (paths, expected_hashes)

        video_key = f"{speaker}.zip"
        video_info = zenodo_files.get(video_key)
        if video_info is None:
            raise ValueError(f"Zenodo metadata has no video archive for {speaker}")
        video_size = int(video_info["size"])
        if video_size != bundle["video_total_size"]:
            raise ValueError(f"video archive size mismatch for {speaker}")
        if video_info.get("checksum") != bundle.get("video_publisher_md5"):
            raise ValueError(f"video publisher checksum metadata mismatch for {speaker}")
        range_start = int(bundle["video_range_start"])
        if range_start + paths["video_metadata_tail"].stat().st_size != video_size:
            raise ValueError(f"video metadata tail is not contiguous through EOF: {speaker}")
        video_codes, _ = zip_codes_from_tail(
            paths["video_metadata_tail"], range_start=range_start, total_size=video_size,
        )
        audio_codes = tar_codes(paths["audio_tar"], ".wav")
        alignment_codes, checked = _read_alignment_codes(paths["alignment_tar"])
        if audio_codes != alignment_codes or audio_codes != video_codes:
            raise ValueError(f"direct source code inventories differ for {speaker}")
        speaker_codes[speaker] = audio_codes
        alignment_counts[speaker] = checked
        source_receipts[speaker] = {
            **expected_urls,
            "audio_tar_sha256": expected_hashes["audio_tar"],
            "alignment_tar_sha256": expected_hashes["alignment_tar"],
            "video_metadata_sha256": expected_hashes["video_metadata_tail"],
            "video_archive_bytes": video_size,
            "video_publisher_md5": video_info.get("checksum"),
            "codes_in_audio_alignment_video": len(audio_codes),
            "alignment_contents_checked": checked,
            "media_payload_decoded": False,
        }

    forbidden_codes = set()
    for path in PINS:
        if path.suffix == ".json":
            forbidden_codes.update(
                row["actual_code"] for row in json.loads(path.read_bytes())
                if row.get("actual_code")
            )
    rows, excluded = build_split_rows(
        speaker_codes, split_speakers, forbidden_codes=forbidden_codes,
    )
    train_speakers = {row["speaker"] for row in rows["train"]}
    validation_speakers = {row["speaker"] for row in rows["validation"]}
    train_content = {normalize_targets(row["targets"]) for row in rows["train"]}
    validation_content = {
        normalize_targets(row["targets"]) for row in rows["validation"]
    }
    if train_speakers & validation_speakers:
        raise ValueError("source speaker overlap after manifest build")
    if train_content & validation_content:
        raise ValueError("normalized target content overlap after manifest build")

    for speaker, (paths, hashes) in source_paths.items():
        if any(digest(path) != hashes[key] for key, path in paths.items()):
            raise ValueError(f"source artifact changed during preparation: {speaker}")
    if digest(bundle_index_path) != bundle_index_sha:
        raise ValueError("bundle index changed during preparation")
    if digest(zenodo_record_path) != zenodo_sha:
        raise ValueError("Zenodo metadata changed during preparation")
    if digest(frozen_summary_path) != frozen_summary_sha:
        raise ValueError("frozen split receipt changed during preparation")
    if digest(frozen_train_path) != frozen_train_sha:
        raise ValueError("frozen train manifest changed during preparation")
    if digest(frozen_validation_path) != frozen_validation_sha:
        raise ValueError("frozen validation manifest changed during preparation")
    for path, expected_sha in PINS.items():
        if digest(path) != expected_sha:
            raise ValueError(f"pinned source changed during preparation: {path}")

    output_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        encoded = json.dumps(rows[split], sort_keys=True, separators=(",", ":")) + "\n"
        (output_dir / f"{split}.json").write_text(encoded, encoding="utf-8", newline="\n")
    summary = {
        "status": "PASS_DIRECT_SOURCE_SPLIT_BUILT",
        "scope": (
            "provisional GRID development; metadata and annotations verified; "
            "media not decoded"
        ),
        "source_record": "doi:10.5281/zenodo.3625687",
        "license_boundary": (
            "CC BY 4.0 per Zenodo metadata; Sheffield describes research use; "
            "internal research only pending product clearance"
        ),
        "frozen_speaker_assignment": split_speakers,
        "counts": {
            "source_speakers": len(speakers),
            "direct_source_assets": sum(map(len, speaker_codes.values())),
            "train": len(rows["train"]),
            "validation": len(rows["validation"]),
            "excluded_known_observed_codes_by_split": excluded,
            "forbidden_code_set_size": len(forbidden_codes),
            "alignment_contents_checked": sum(alignment_counts.values()),
        },
        "integrity": {
            "all_speaker_direct_audio_alignment_video_inventories_match": True,
            "all_alignment_contents_match_decoded_codes": True,
            "train_validation_speaker_overlap": 0,
            "train_validation_asset_identity_overlap": 0,
            "train_validation_code_overlap": 0,
            "train_validation_normalized_target_overlap": 0,
            "known_observed_code_overlap": 0,
            "media_payload_decoded": False,
            "media_content_duplicate_scan": False,
            "cross_corpus_physical_speaker_independence_verified": False,
        },
        "frozen_reference": {
            "train_sha256": frozen_train_sha,
            "validation_sha256": frozen_validation_sha,
            "summary_sha256": frozen_summary_sha,
        },
        "source_receipts": source_receipts,
        "bundle_index_sha256": bundle_index_sha,
        "zenodo_record_sha256": zenodo_sha,
        "pinned_exclusion_inputs": expected_pins,
        "output_sha256": {
            split: digest(output_dir / f"{split}.json")
            for split in ("train", "validation")
        },
        "builder_sha256": digest(Path(__file__)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-index", type=Path, required=True)
    parser.add_argument("--zenodo-record", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--frozen-train", type=Path, required=True)
    parser.add_argument("--frozen-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_direct_source_split(
        bundle_index_path=args.bundle_index,
        zenodo_record_path=args.zenodo_record,
        frozen_summary_path=args.frozen_summary,
        frozen_train_path=args.frozen_train,
        frozen_validation_path=args.frozen_validation,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
