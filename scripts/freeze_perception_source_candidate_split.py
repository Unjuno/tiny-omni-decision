"""Freeze a group-disjoint development split over locally available train media."""

import argparse
import hashlib
import json
from pathlib import Path


def _rank(seed, value):
    return hashlib.sha256(f"perception-source-candidate-dev-v1:{seed}:{value}".encode()).hexdigest()


def freeze_split(records, media_clips, *, validation_group_count, seed):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    by_video = {}
    for video_id, clip in media_clips.items():
        if clip.get("metadata", {}).get("split") != "train":
            raise ValueError("only official train media may enter the development split")
        asset_hash = clip.get("sha256")
        if not isinstance(asset_hash, str) or len(asset_hash) != 64:
            raise ValueError("every media asset must have a SHA-256 receipt")
        by_video[video_id] = asset_hash
    if len(set(by_video.values())) != len(by_video):
        raise ValueError("duplicate media asset SHA-256 must be grouped before splitting")

    candidates_by_video = {}
    for record in records:
        if record.get("status") != "candidate":
            continue
        video_id = record.get("group_id")
        if record.get("split") != "train":
            raise ValueError("candidate corpus contains a non-train record")
        if record.get("media_reference", {}).get("video_id") != video_id:
            raise ValueError("media reference and video grouping identity differ")
        if video_id not in by_video:
            continue
        cut_frame = record.get("provenance", {}).get("upstream_cut_frame")
        metadata = media_clips[video_id]["metadata"]
        if type(cut_frame) is not int or not -1 <= cut_frame < metadata["num_frames"]:
            raise ValueError("candidate record lacks valid upstream cut provenance")
        if cut_frame != media_clips[video_id].get("cut_frame"):
            raise ValueError("candidate and media receipt cut-frame mappings differ")
        cut_duration_us = (
            metadata["num_frames"] / metadata["frame_rate"] * 1e6
            if cut_frame == -1
            else cut_frame / metadata["frame_rate"] * 1e6
        )
        audio_duration_us = metadata["audio_samples"] / metadata["audio_sample_rate"] * 1e6
        media_limit_us = min(
            cut_duration_us,
            metadata["num_frames"] / metadata["frame_rate"] * 1e6,
            audio_duration_us,
        )
        if record["request"]["audio_interval_us"][1] > media_limit_us:
            raise ValueError("candidate interval extends past media or upstream cut boundary")
        frame_ids = [
            frame["frame_id"]
            for candidate in record["request"]["candidates"]
            for frame in candidate["frame_boxes"]
        ]
        if cut_frame >= 0 and any(frame_id >= cut_frame for frame_id in frame_ids):
            raise ValueError("candidate visual proposal extends past upstream cut frame")
        candidates_by_video.setdefault(video_id, []).append(record)
    video_ids = sorted(candidates_by_video)
    if not 1 <= validation_group_count < len(video_ids):
        raise ValueError("validation group count must be between 1 and group_count - 1")

    ranked_video_ids = sorted(video_ids, key=lambda video_id: _rank(seed, video_id))
    validation_video_ids = sorted(ranked_video_ids[:validation_group_count])
    validation_set = set(validation_video_ids)
    train_video_ids = sorted(set(video_ids) - validation_set)
    train_records = sorted(
        (record for video_id in train_video_ids for record in candidates_by_video[video_id]),
        key=lambda record: record["record_id"],
    )
    validation_records = sorted(
        (record for video_id in validation_video_ids for record in candidates_by_video[video_id]),
        key=lambda record: record["record_id"],
    )
    train_assets = sorted(by_video[video_id] for video_id in train_video_ids)
    validation_assets = sorted(by_video[video_id] for video_id in validation_video_ids)
    if (
        set(train_video_ids) & set(validation_video_ids)
        or set(train_assets) & set(validation_assets)
    ):
        raise AssertionError("split integrity violation")
    return {
        "seed": seed,
        "train_video_ids": train_video_ids,
        "validation_video_ids": validation_video_ids,
        "train_asset_sha256": train_assets,
        "validation_asset_sha256": validation_assets,
        "train_records": train_records,
        "validation_records": validation_records,
        "candidate_group_count": len(video_ids),
        "candidate_example_count": len(train_records) + len(validation_records),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_files(
    corpus_path, acquisition_manifest_path, output_dir, *, validation_group_count, seed
):
    corpus_path = Path(corpus_path)
    acquisition_manifest_path = Path(acquisition_manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior split: {output_dir}")
    records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    acquisition = json.loads(acquisition_manifest_path.read_text(encoding="utf-8"))
    media_clips = {
        clip["metadata"]["video_id"]: clip
        for clip in acquisition["clips"]
    }
    split = freeze_split(
        records,
        media_clips,
        validation_group_count=validation_group_count,
        seed=seed,
    )
    output_dir.mkdir(parents=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    for path, rows in (
        (train_path, split["train_records"]),
        (validation_path, split["validation_records"]),
    ):
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "format": "perception_test_source_candidate_grouped_dev_split_v1",
        "source_split": "official train only",
        "seed": seed,
        "group_policy": "video_id; exact media SHA-256 duplicates rejected before split",
        "selection_policy": "SHA-256 rank by fixed seed; no metric-based selection",
        "validation_is_blind": False,
        "participant_identity_grouping": "UNKNOWN",
        "normalized_content_overlap": "NOT_CHECKED",
        "candidate_group_count": split["candidate_group_count"],
        "train_video_group_count": len(split["train_video_ids"]),
        "validation_video_group_count": len(split["validation_video_ids"]),
        "train_example_count": len(split["train_records"]),
        "validation_example_count": len(split["validation_records"]),
        "video_id_overlap_count": len(
            set(split["train_video_ids"]) & set(split["validation_video_ids"])
        ),
        "media_sha256_overlap_count": len(
            set(split["train_asset_sha256"]) & set(split["validation_asset_sha256"])
        ),
        "train_video_ids": split["train_video_ids"],
        "validation_video_ids": split["validation_video_ids"],
        "train_jsonl": train_path.name,
        "train_jsonl_sha256": _sha256(train_path),
        "validation_jsonl": validation_path.name,
        "validation_jsonl_sha256": _sha256(validation_path),
        "source_corpus_sha256": _sha256(corpus_path),
        "acquisition_manifest_sha256": _sha256(acquisition_manifest_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--acquisition-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-groups", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_files(
                args.corpus,
                args.acquisition_manifest,
                args.output_dir,
                validation_group_count=args.validation_groups,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
