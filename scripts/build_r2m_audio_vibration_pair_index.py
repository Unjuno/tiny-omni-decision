"""Freeze the R2-M pair index from allowed trial metadata only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import h5py

from tiny_omni_decision.r2m_pairs import build_audio_vibration_pairs

PARTICIPANT_SPLIT_SHA256 = "0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0"
DOWNLOAD_INDEX_SHA256 = "715dd57fdf1967767f28f2b2246c57232dfba3e74ea548d46c83e68b882402d4"
SAMPLE_RATE_HZ = 16000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def participant_splits(manifest: dict[str, object]) -> dict[int, str]:
    declared_hash = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items()
               if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    if (declared_hash != PARTICIPANT_SPLIT_SHA256
            or hashlib.sha256(encoded).hexdigest() != declared_hash):
        raise ValueError("unexpected frozen participant split manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("participant split manifest is missing splits")
    result: dict[int, str] = {}
    for name in ("train", "calibration", "validation", "sealed_audit"):
        ids = splits.get(name)
        if not isinstance(ids, list):
            raise ValueError(f"participant split is missing {name}")
        for participant in ids:
            if type(participant) is not int or participant in result:
                raise ValueError("invalid or duplicate participant assignment")
            result[participant] = name
    if set(result) != {1, 2, 4, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17}:
        raise ValueError("participant inventory differs from frozen split v2")
    if 0 in result or 999 in result:
        raise ValueError("quarantine/sentinel participant must not enter split map")
    return result


def collect_allowed_records(
    source_dir: Path,
    receipt: dict[str, object],
    participant_to_split: dict[int, str],
) -> list[dict[str, object]]:
    files = receipt.get("files")
    if not isinstance(files, list):
        raise ValueError("payload receipt has no file list")
    records: list[dict[str, object]] = []
    for item in files:
        if not isinstance(item, dict) or item.get("verification") != "PASS":
            raise ValueError("payload verification receipt contains a failed item")
        name = item.get("file")
        if not isinstance(name, str) or not name.endswith(".zip"):
            raise ValueError("unexpected payload archive name")
        if name == "00_no-activity.zip":
            continue
        archive = source_dir / name
        if archive.stat().st_size != item.get("bytes"):
            raise ValueError(f"archive size differs from verified receipt: {name}")
        actual_sha = sha256_file(archive)
        if actual_sha != item.get("sha256"):
            raise ValueError(f"archive SHA-256 differs from verified receipt: {name}")

        activity = name[:-4]
        with zipfile.ZipFile(archive) as zf:
            members = [path for path in zf.namelist()
                       if path.endswith("_multisensor-1.hdf5")]
            if len(members) != 1:
                raise ValueError(f"expected one multisensor-1 member in {name}")
            raw_hdf5 = zf.read(members[0])
        with h5py.File(io.BytesIO(raw_hdf5), "r") as data:
            participant_ids = data["participant"][:]
            trial_ids = data["trial"][:]
            if len(participant_ids) != len(trial_ids):
                raise ValueError(f"participant/trial row count differs in {name}")
            for index, (raw_pid, raw_trial) in enumerate(
                zip(participant_ids, trial_ids, strict=True)
            ):
                participant_id = int(raw_pid)
                split = participant_to_split.get(participant_id)
                if split not in {"train", "calibration", "validation"}:
                    # Do not index or dereference any signal dataset for sealed,
                    # quarantined, sentinel, or otherwise unassigned rows.
                    continue
                audio_samples = len(data["audio"][index])
                axes = [data[f"vibration-{axis}"][index] for axis in "xyz"]
                vibration_samples = len(axes[0])
                if (audio_samples <= 0 or vibration_samples <= 0
                        or any(len(axis) != vibration_samples for axis in axes)):
                    raise ValueError(f"empty or inconsistent paired signals in {name}")
                vibration_rate = float(data["vibration_sample_rate"][index])
                if not math.isfinite(vibration_rate) or vibration_rate <= 0:
                    raise ValueError(f"invalid vibration sample rate in {name}")
                ratio = ((audio_samples / SAMPLE_RATE_HZ)
                         / (vibration_samples / vibration_rate))
                if not math.isfinite(ratio) or abs(ratio - 1.0) > 0.005:
                    raise ValueError(f"audio/vibration duration mismatch in {name}")
                records.append({
                    "activity": activity,
                    "participant_id": participant_id,
                    "trial_id": int(raw_trial),
                    "split": split,
                    "audio_samples": audio_samples,
                    "audio_sample_rate_hz": SAMPLE_RATE_HZ,
                    "vibration_samples": vibration_samples,
                    "vibration_sample_rate_hz": vibration_rate,
                    "duration_ratio_audio_to_vibration": ratio,
                    "source_archive": name,
                    "source_member": members[0],
                })
        allowed_count = sum(r["source_archive"] == name for r in records)
        print(
            f"verified and indexed {name}: {allowed_count} allowed records",
            flush=True,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(
        "manifests/pretrained_reuse/r2m-audio-vibration-split-v2-20261120.json"
    )
    receipt_path = Path(
        "manifests/pretrained_reuse/"
        "r2m-audio-vibration-payload-verification-20261120.json"
    )
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("download_index_sha256") != DOWNLOAD_INDEX_SHA256:
        raise SystemExit("unexpected source download-index digest")
    rows = collect_allowed_records(
        args.artifact_root / "source", receipt, participant_splits(split_manifest)
    )
    pairs = build_audio_vibration_pairs(
        rows, max_duration_delta=args.duration_tolerance
    )
    (args.output_dir / "allowed-trial-metadata.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "pair-index.json").write_text(
        json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "input_trials_by_split": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "calibration", "validation")
        },
        "pair_count_by_split": pairs["pair_count_by_split"],
        "negative_relation_count": pairs["negative_relation_count"],
        "unpaired_trial_count_by_split": pairs["unpaired_trial_count_by_split"],
        "pair_index_sha256": pairs["manifest_sha256"],
        "sealed_audit_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
