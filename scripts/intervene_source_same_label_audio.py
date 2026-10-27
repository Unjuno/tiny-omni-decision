"""Evaluate same-annotated-label audio substitution on saved train-only CV heads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.replay_source_unimodal_cv import RUN_SHA, compare_prediction
from scripts.summarize_source_unimodal_cv import paired_interval
from scripts.train_source_unimodal_cv import (
    AUDIO,
    LABELS,
    PINNED,
    PLAN,
    SOURCE,
    SOURCE_RUNNER,
    VIDEO,
    CandidateModel,
    digest,
    fold_assignment,
)

ARCHIVE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip"
)
ARCHIVE_SHA = "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"


def same_label_donors(entries):
    groups = {}
    for row in entries:
        if row["label"] is not None:
            groups.setdefault((row["fold"], row["label"]), {}).setdefault(
                row["video_id"], []
            ).append(row["record_id"])
    result = {}
    for videos in groups.values():
        ordered = sorted(videos)
        if len(ordered) < 2:
            continue
        for index, video in enumerate(ordered):
            donor = min(videos[ordered[(index + 1) % len(ordered)]])
            result.update({record: donor for record in videos[video]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output exists")
    if digest(args.run_dir / "run.json") != RUN_SHA or digest(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("run or source archive hash mismatch")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    run = json.loads((args.run_dir / "run.json").read_bytes())
    label_rows = [json.loads(line) for line in LABELS.read_bytes().splitlines()]
    if len({row["record_id"] for row in label_rows}) != 2593:
        raise ValueError("unexpected source record IDs")
    with zipfile.ZipFile(ARCHIVE) as archive:
        source = json.loads(archive.read("all_train.json"))
    sound_lookup = {}
    for video_id in {row["group_id"] for row in label_rows}:
        events = source[video_id]["sound_localisation"]
        if len({event["id"] for event in events}) != len(events):
            raise ValueError("duplicate source event identity")
        sound_lookup[video_id] = {event["id"]: event["label_id"] for event in events}
    del source
    assignments = fold_assignment([row["group_id"] for row in label_rows])
    if (
        hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()
        != run["assignments_sha256"]
    ):
        raise ValueError("fold assignment mismatch")
    entries, unmatched_reasons = [], Counter()
    for row in label_rows:
        video = row["group_id"]
        event_ids = row["provenance"]["coalesced_sound_annotation_ids"]
        labels = {sound_lookup[video].get(event_id) for event_id in event_ids}
        label = next(iter(labels)) if len(labels) == 1 and None not in labels else None
        if label is None:
            unmatched_reasons["missing_or_mixed_source_labels"] += 1
        entries.append(
            {
                "record_id": row["record_id"],
                "video_id": video,
                "label": label,
                "fold": assignments[video],
            }
        )
    donors = same_label_donors(entries)
    unmatched_reasons["no_other_video_same_label_in_fold"] = sum(
        row["label"] is not None and row["record_id"] not in donors for row in entries
    )
    entry_by_id = {row["record_id"]: row for row in entries}
    for recipient, donor in donors.items():
        a, b = entry_by_id[recipient], entry_by_id[donor]
        if a["video_id"] == b["video_id"] or a["label"] != b["label"] or a["fold"] != b["fold"]:
            raise ValueError("invalid same-label donor")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source_readout", SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    examples, _ = module.load_examples(PLAN, LABELS, VIDEO, AUDIO)
    by_id = {e.record_id: e for e in examples}
    if len(by_id) != 2593:
        raise ValueError("feature record mismatch")
    args.output_dir.mkdir(parents=True)
    records = []
    for fold in range(3):
        entry = next(e for e in run["results"] if e["fold"] == fold and e["mode"] == "joint")
        checkpoint = args.run_dir / f"fold-{fold}-joint.pt"
        predictions = args.run_dir / f"fold-{fold}-joint-predictions.jsonl"
        if (
            digest(checkpoint) != entry["checkpoint_sha256"]
            or digest(predictions) != entry["prediction_sha256"]
        ):
            raise ValueError("checkpoint or prediction hash mismatch")
        saved = {
            row["record_id"]: row for row in map(json.loads, predictions.read_bytes().splitlines())
        }
        eligible = [
            e for e in examples if assignments[e.video_id] == fold and e.record_id in donors
        ]
        model = CandidateModel()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(eligible), 32):
                indices = list(range(start, min(start + 32, len(eligible))))
                video, audio, targets, mask, batch = module.collate_examples(eligible, indices)
                mismatch = torch.from_numpy(
                    np.stack([by_id[donors[e.record_id]].audio for e in batch])
                )
                outputs = {}
                for condition, candidate_audio in [
                    ("original", audio),
                    ("same_label_other_video", mismatch),
                ]:
                    logits = model(video, candidate_audio, mask, "joint")
                    prob = torch.softmax(logits, dim=1)
                    outputs[condition] = (
                        prob,
                        module.grouped_nll(logits, targets),
                        prob.argmax(dim=1),
                    )
                for index, example in enumerate(batch):
                    base, base_nll, base_choice = outputs["original"]
                    dist = targets[index].float() / targets[index].sum()
                    probs = base[index, : len(example.candidate_ids)].tolist() + [
                        float(base[index, -1])
                    ]
                    compare_prediction(
                        saved[example.record_id],
                        example,
                        probs,
                        bool(targets[index, base_choice[index]]),
                        float(base_nll[index]),
                        float((base[index] - dist).square().sum()),
                    )
                    row = {
                        "record_id": example.record_id,
                        "video_id": example.video_id,
                        "fold": fold,
                        "label": entry_by_id[example.record_id]["label"],
                        "none_visible": example.none_visible,
                    }
                    for condition, (_prob, nll, chosen) in outputs.items():
                        row[condition] = {
                            "correct": bool(targets[index, chosen[index]]),
                            "nll": float(nll[index]),
                        }
                    records.append(row)
    if len(records) != len(donors):
        raise ValueError("matched record count mismatch")
    output = args.output_dir / "predictions.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    deltas = {}
    for field in ("correct", "nll"):
        deltas[field] = paired_interval(
            [
                {
                    "video_id": row["video_id"],
                    "delta": float(row["original"][field])
                    - float(row["same_label_other_video"][field]),
                }
                for row in records
            ]
        )
    result = {
        "status": "PASS_SAME_LABEL_TRAIN_ONLY_OOF_DIAGNOSTIC",
        "source_run_sha256": RUN_SHA,
        "source_archive_sha256": ARCHIVE_SHA,
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": hashlib.sha256(
            Path("docs/pretrained_reuse/R2_SOURCE_SAME_LABEL_AUDIO_PROTOCOL_20261027.md")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
        "donor_mapping_sha256": hashlib.sha256(
            json.dumps(donors, sort_keys=True).encode()
        ).hexdigest(),
        "predictions_sha256": digest(output),
        "total_records": len(entries),
        "matched_records": len(records),
        "unmatched_reasons": dict(unmatched_reasons),
        "coverage_by_label": dict(Counter(str(row["label"]) for row in records)),
        "coverage_by_fold": dict(Counter(str(row["fold"]) for row in records)),
        "accuracy": {
            condition: float(np.mean([row[condition]["correct"] for row in records]))
            for condition in ("original", "same_label_other_video")
        },
        "strata": {
            name: {
                "records": len([r for r in records if r["none_visible"] == flag]),
                **{
                    condition: float(
                        np.mean(
                            [r[condition]["correct"] for r in records if r["none_visible"] == flag]
                        )
                    )
                    for condition in ("original", "same_label_other_video")
                },
            }
            for name, flag in [("visible", False), ("none_visible", True)]
        },
        "paired_original_minus_matched": deltas,
        "seed": 17,
        "bootstrap_repetitions": 10000,
        "training": False,
        "old_development_or_sealed_audit_opened": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "limitations": "same source label is not equivalent to same physical source or session",
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
