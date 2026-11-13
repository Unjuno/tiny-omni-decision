"""Freeze a label-free feature plan for the fixed official-validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.prepare_perception_source_candidate_features import (
        build_validation_feature_plan,
    )
except ModuleNotFoundError:
    from prepare_perception_source_candidate_features import build_validation_feature_plan


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--media-report", type=Path, required=True)
    parser.add_argument("--media-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation feature plan: {args.output}")
    corpus_rows = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [row for row in corpus_rows if row.get("status") == "candidate"]
    corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    acquisition = json.loads(args.acquisition_manifest.read_text(encoding="utf-8"))
    report = json.loads(args.media_report.read_text(encoding="utf-8"))
    media_validation = json.loads(args.media_validation.read_text(encoding="utf-8"))
    corpus_hash = sha256_file(args.corpus)
    if (
        corpus_hash != corpus_manifest.get("record_file_sha256")
        or corpus_manifest.get("source_split") != "official_validation"
        or corpus_manifest.get("selection_sha256") != report.get("selection_sha256")
        or corpus_manifest.get("selection_sha256") != media_validation.get("selection_sha256")
        or len(records) != corpus_manifest.get("eligible_candidate_count")
        or len(records) != 138
    ):
        raise ValueError("fixed official-validation corpus provenance/count mismatch")
    clips = {}
    for clip in acquisition.get("clips", []):
        video_id = clip.get("metadata", {}).get("video_id")
        if not video_id or video_id in clips:
            raise ValueError("derived acquisition manifest has missing/duplicate video IDs")
        clips[video_id] = clip
    plan = build_validation_feature_plan(
        records,
        clips,
        report,
        record_jsonl_sha256=corpus_hash,
        acquisition_manifest_sha256=sha256_file(args.acquisition_manifest),
        media_report_sha256=sha256_file(args.media_report),
        selection_sha256=corpus_manifest["selection_sha256"],
    )
    plan["source_revision"] = corpus_manifest["source_revision"]
    plan["corpus_manifest_sha256"] = sha256_file(args.corpus_manifest)
    plan["media_validation_sha256"] = sha256_file(args.media_validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS_LABEL_FREE_VALIDATION_PLAN",
                "records": plan["validation"]["record_count"],
                "video_groups": plan["validation"]["video_group_count"],
                "plan_sha256": sha256_file(args.output),
                "target_labels_serialized": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
