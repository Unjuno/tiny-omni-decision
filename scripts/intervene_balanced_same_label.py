"""Same-label other-video audio intervention on fixed balanced set models."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.intervene_source_same_label_audio import same_label_donors
from scripts.source_set_decision import decode_set
from scripts.source_set_metrics import score_record
from scripts.summarize_source_temporal_long_curve import paired_bootstrap
from scripts.train_source_unimodal_cv import (
    AUDIO,
    LABELS,
    PINNED,
    PLAN,
    ROOT,
    SOURCE,
    SOURCE_RUNNER,
    VIDEO,
    CandidateModel,
    digest,
    fold_assignment,
)

RUN_SHA = "ee04092410d024692b8123f953bc65c772fffb8f930db03f5a72e633d216267c"
DONOR_RUN_SHA = "30471d27f3c9dff4ba6427dbc5fe734bda97927a5b55ec554733252e64ed8b24"
PROTOCOL = Path("docs/pretrained_reuse/R2_BALANCED_SAME_LABEL_AUDIO_PROTOCOL_20261027.md")


def result_row(example, logits, index, mask):
    ids = list(example.candidate_ids)
    selected = decode_set(logits, mask)[index]
    choices = [ids[k] for k in range(len(ids)) if selected[k]]
    if selected[-1]:
        choices.append("none_visible")
    conditional = torch.sigmoid(logits[index, : len(ids)]).tolist()
    none = float(torch.sigmoid(logits[index, -1]))
    metrics = score_record(ids, sorted(example.target_candidate_ids), choices, conditional, none)
    return {
        "selected": choices,
        "conditional_probabilities": conditional,
        "none_probability": none,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--donor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("output exists or outside artifact root")
    if shutil.disk_usage(ROOT).free < 2**30 + 32 * 2**20:
        raise OSError("insufficient disk reserve")
    if (
        digest(args.run_dir / "run.json") != RUN_SHA
        or digest(args.donor_dir / "result.json") != DONOR_RUN_SHA
    ):
        raise ValueError("run receipt mismatch")
    donor_result = json.loads((args.donor_dir / "result.json").read_bytes())
    donor_file = args.donor_dir / "predictions.jsonl"
    if digest(donor_file) != donor_result["predictions_sha256"]:
        raise ValueError("donor records mismatch")
    donor_rows = list(map(json.loads, donor_file.read_bytes().splitlines()))
    donors = same_label_donors(donor_rows)
    if (
        len(donors) != 2581
        or hashlib.sha256(json.dumps(donors, sort_keys=True).encode()).hexdigest()
        != donor_result["donor_mapping_sha256"]
    ):
        raise ValueError("donor mapping mismatch")
    source_rows = {r["record_id"]: r for r in donor_rows}
    for recipient, donor in donors.items():
        left, right = source_rows[recipient], source_rows[donor]
        if (
            left["video_id"] == right["video_id"]
            or left["fold"] != right["fold"]
            or left["label"] != right["label"]
        ):
            raise ValueError("invalid donor")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input mismatch {path}")
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_hash != "ef3d1021d7c17fdffac68f45fbd2f40778d247e466c4f90fbc0e8c447a5ce2c0":
        raise ValueError("protocol hash mismatch")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.set_num_threads(4)
    examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
    by_id = {e.record_id: e for e in examples}
    groups = fold_assignment([e.video_id for e in examples])
    run = json.loads((args.run_dir / "run.json").read_bytes())
    if (
        len(by_id) != 2593
        or hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
        != run["assignment_sha256"]
    ):
        raise ValueError("feature/group mismatch")
    args.output_dir.mkdir(parents=True)
    records = []
    for fold in range(3):
        if shutil.disk_usage(ROOT).free < 2**30:
            raise OSError("disk reserve exhausted")
        meta = next(r for r in run["results"] if r["fold"] == fold and r["mode"] == "joint")
        checkpoint = args.run_dir / f"fold-{fold}-joint.pt"
        predictions = args.run_dir / f"fold-{fold}-joint-predictions.jsonl"
        if (
            digest(checkpoint) != meta["checkpoint_sha256"]
            or digest(predictions) != meta["prediction_sha256"]
        ):
            raise ValueError("checkpoint/prediction mismatch")
        saved = {r["record_id"]: r for r in map(json.loads, predictions.read_bytes().splitlines())}
        eligible = [e for e in examples if groups[e.video_id] == fold and e.record_id in donors]
        model = CandidateModel()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        with torch.inference_mode():
            for pos in range(0, len(eligible), 32):
                indices = list(range(pos, min(pos + 32, len(eligible))))
                video, audio, _, mask, batch = source.collate_examples(eligible, indices)
                donor_audio = torch.from_numpy(
                    np.stack([by_id[donors[e.record_id]].audio for e in batch])
                )
                original_logits = model(video, audio, mask, "joint")
                replacement_logits = model(video, donor_audio, mask, "joint")
                for j, example in enumerate(batch):
                    original = result_row(example, original_logits, j, mask)
                    swapped = result_row(example, replacement_logits, j, mask)
                    expected = saved[example.record_id]
                    if (
                        original["selected"] != expected["selected"]
                        or abs(original["none_probability"] - expected["none_probability"]) > 1e-6
                        or len(original["conditional_probabilities"])
                        != len(expected["conditional_probabilities"])
                        or any(
                            abs(a - b) > 1e-6
                            for a, b in zip(
                                original["conditional_probabilities"],
                                expected["conditional_probabilities"],
                                strict=True,
                            )
                        )
                    ):
                        raise ValueError("original prediction replay mismatch")
                    n = len(example.target_candidate_ids)
                    records.append(
                        {
                            "record_id": example.record_id,
                            "video_id": example.video_id,
                            "fold": fold,
                            "label": source_rows[example.record_id]["label"],
                            "donor_record_id": donors[example.record_id],
                            "stratum": "none" if n == 0 else "one" if n == 1 else "multi",
                            "original": original,
                            "replacement": swapped,
                        }
                    )
    if len(records) != len(donors) or len({r["record_id"] for r in records}) != len(donors):
        raise ValueError("matched coverage mismatch")
    out = args.output_dir / "predictions.jsonl"
    with out.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    report = {}
    for group in ("all", "none", "one", "multi"):
        subset = [r for r in records if group == "all" or r["stratum"] == group]
        report[group] = {"n": len(subset)}
        for field in ("exact", "membership_nll", "membership_brier", "none_nll", "none_brier"):
            pairs = [
                (dict(r["replacement"], video_id=r["video_id"]), r["original"]) for r in subset
            ]
            report[group][field] = {
                "original": float(np.mean([r["original"][field] for r in subset])),
                "replacement": float(np.mean([r["replacement"][field] for r in subset])),
                "original_minus_replacement": paired_bootstrap(pairs, field),
            }
    result = {
        "status": "PASS_SAME_LABEL_INTERVENTION",
        "run_sha256": RUN_SHA,
        "donor_result_sha256": DONOR_RUN_SHA,
        "donor_mapping_sha256": donor_result["donor_mapping_sha256"],
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": protocol_hash,
        "predictions_sha256": digest(out),
        "matched": len(records),
        "unmatched": 12,
        "report": report,
        "bootstrap_draws": 10000,
        "seed": 17,
        "training": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "sealed_audit_opened": False,
        "scope": "nonblind train-source diagnostic; same label does not prove same physical source",
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
