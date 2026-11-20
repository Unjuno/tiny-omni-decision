"""Freeze a label-blind recording-ID split for SoundActions v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

FORMAT = "soundactions_recording_split_v1"
SEED = 17
SEEN_IDS = tuple(range(1, 36))
SAMPLE_IDS = tuple(range(1, 366))
MEDIA_PATH = re.compile(r"^(JPG|MP4|WAV)/SoundAction (\d{3}) - .+\.(jpg|mp4|wav)$")


def _bucket(sample_id: int) -> float:
    payload = f"soundactions-v2|seed={SEED}|recording={sample_id:03d}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64


def build_manifest() -> dict[str, object]:
    splits: dict[str, list[int]] = {name: [] for name in ("train", "validation", "sealed_audit")}
    for sample_id in SAMPLE_IDS:
        if sample_id in SEEN_IDS:
            split = "train"
        else:
            bucket = _bucket(sample_id)
            split = "train" if bucket < 0.70 else "validation" if bucket < 0.85 else "sealed_audit"
        splits[split].append(sample_id)

    payload: dict[str, object] = {
        "format": FORMAT,
        "source": {
            "name": "365 Sound Actions",
            "record_id": "zenodo:23971067",
            "version": 2,
            "license": "CC BY 4.0",
        },
        "task": "three-way Decision over impulsive, sustained, iterative perception_type",
        "seed": SEED,
        "assignment": {
            "hash_input_template": "soundactions-v2|seed=17|recording={recording_id:03d}",
            "bucket": "int.from_bytes(sha256(input).digest()[:8], 'big') / 2^64",
            "thresholds": {"train": "<0.70", "validation": "[0.70,0.85)",
                           "sealed_audit": ">=0.85"},
        },
        "known_exposure": {
            "reason": (
                "Official overview preview exposed label and title rows 1-35 "
                "before split freeze."
            ),
            "recording_ids_forced_to_train": list(SEEN_IDS),
            "no_other_per_record_label_values_used_for_assignment": True,
        },
        "grouping": {
            "unit": "recording_id",
            "modalities_together": ["audio", "video", "still"],
            "same_recordist_across_all_records": True,
            "person_generalization_claim": False,
        },
        "media_identity_overlap_check": "pending-after-acquisition",
        "sealed_audit_accessed": False,
        "splits": splits,
        "counts": {name: len(values) for name, values in splits.items()},
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_publisher_media_checksums(path: Path, splits: dict[str, list[int]]) -> dict[str, object]:
    """Check exact media-byte hashes against the source checksum inventory."""
    sample_split = {
        sample_id: split for split, ids in splits.items() for sample_id in ids
    }
    expected_keys = {
        (modality, sample_id)
        for sample_id in sample_split
        for modality in ("JPG", "MP4", "WAV")
    }
    seen: dict[tuple[str, int], str] = {}
    hashes: dict[tuple[str, str], list[int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative_path = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"malformed publisher checksum line {line_number}") from error
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid publisher SHA-256 on line {line_number}")
        match = MEDIA_PATH.fullmatch(relative_path)
        if not match:
            continue
        modality, raw_id, extension = match.groups()
        if {"JPG": "jpg", "MP4": "mp4", "WAV": "wav"}[modality] != extension:
            raise ValueError(f"media extension differs from archive path on line {line_number}")
        sample_id = int(raw_id)
        if sample_id not in sample_split:
            continue
        key = (modality, sample_id)
        if key in seen:
            raise ValueError(f"duplicate publisher checksum path for {key}")
        seen[key] = digest
        hashes.setdefault((modality, digest), []).append(sample_id)

    if set(seen) != expected_keys:
        missing = len(expected_keys - set(seen))
        extra = len(set(seen) - expected_keys)
        raise ValueError(f"publisher checksum inventory mismatch: missing={missing}, extra={extra}")
    collisions = [ids for ids in hashes.values() if len(ids) > 1]
    if collisions:
        raise ValueError("identical publisher media hashes occur across recording IDs")
    return {
        "status": "PASS_EXACT_BYTES",
        "publisher_checksum_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recordings_checked": len(sample_split),
        "media_files_checked": len(seen),
        "cross_recording_byte_hash_collisions": 0,
        "scope": "publisher-listed exact byte identity; not perceptual or semantic overlap",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checksums", type=Path)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite split manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest()
    if args.checksums is not None:
        payload["media_integrity"] = verify_publisher_media_checksums(
            args.checksums, payload["splits"]
        )
        payload["media_identity_overlap_check"] = "PASS_EXACT_BYTES"
        payload.pop("manifest_sha256", None)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "manifest_sha256": payload["manifest_sha256"],
                      "counts": payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
