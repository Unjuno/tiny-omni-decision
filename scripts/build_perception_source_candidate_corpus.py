"""Build a deterministic train-only Perception Test source-candidate corpus."""

import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

from tiny_omni_decision.perception_source_candidate import source_candidate_records

SOURCE_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
EXPECTED_TRAIN_SHA256 = "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(input_zip, output_dir):
    input_zip = Path(input_zip)
    output_dir = Path(output_dir)
    source_hash = sha256(input_zip)
    if source_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError("train annotation archive hash differs from the pinned source")
    with zipfile.ZipFile(input_zip) as archive:
        names = archive.namelist()
        if "all_train.json" not in names:
            raise ValueError("pinned source archive must contain all_train.json")
        source = json.loads(archive.read("all_train.json"))

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "source_candidate_records.jsonl"
    counts = collections.Counter()
    candidate_videos = set()
    candidate_counts = collections.Counter()
    positive_counts = collections.Counter()
    event_label_counts = collections.Counter()
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for row in sorted(source.values(), key=lambda item: item["metadata"]["video_id"]):
            video_id = row["metadata"]["video_id"]
            sounds = {sound["id"]: sound for sound in row.get("sound_localisation", [])}
            for record in source_candidate_records(video_id, row):
                counts[record["status"]] += 1
                if record["status"] == "candidate":
                    candidate_videos.add(video_id)
                    candidate_counts[len(record["request"]["candidates"])] += 1
                    positive_counts[len(record["supervision"]["target_candidate_ids"])] += 1
                    sound_id = int(record["record_id"].rsplit(":", 1)[1])
                    event_label_counts[sounds[sound_id]["label_id"]] += 1
                output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "format": "perception_test_audio_visual_source_candidate_v1",
        "source_revision": SOURCE_REVISION,
        "source_archive_sha256": source_hash,
        "adapter_sha256": sha256(
            Path(__file__).parents[1]
            / "src"
            / "tiny_omni_decision"
            / "perception_source_candidate.py"
        ),
        "builder_sha256": sha256(__file__),
        "source_member": "all_train.json",
        "split_policy": "official train only; all events grouped by video_id",
        "row_count": sum(counts.values()),
        "status_counts": dict(sorted(counts.items())),
        "candidate_video_count": len(candidate_videos),
        "candidate_examples": counts["candidate"],
        "candidate_count_distribution": dict(sorted(candidate_counts.items())),
        "positive_candidate_count_distribution": dict(sorted(positive_counts.items())),
        "source_label_id_counts_metadata_only": dict(sorted(event_label_counts.items())),
        "record_file": destination.name,
        "record_file_sha256": sha256(destination),
        "label_ids_are_not_runtime_inputs": True,
        "validation_or_audit_loaded": False,
        "media_decoded": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = build(args.input_zip, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
