"""One-time CPU transfer check on CLEVR-4 assets unseen by the selected fit checkpoint."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.native_fit import summarize  # noqa: E402
from scripts.native_fit_runtime import FeatureStore  # noqa: E402
from scripts.native_video_cache import sha  # noqa: E402
from scripts.replay_native_common_cpu import (  # noqa: E402
    load_json,
    score_validation,
    verify_file,
    verify_inputs,
)
from tiny_omni_decision.native_candidate import NativeCandidateReadout  # noqa: E402

CONFIG = REPO / "configs/pretrained_reuse/native_common_fit_v1.json"


def unused_image_rows(train_rows, order, selected_step):
    if (
        type(selected_step) is not int
        or not 0 < selected_step <= len(order)
        or any(type(i) is not int or not 0 <= i < len(train_rows) for batch in order for i in batch)
    ):
        raise ValueError("valid sampler order and selected checkpoint step required")
    seen_ids = {train_rows[i]["id"] for batch in order[:selected_step] for i in batch}
    seen_groups = {
        train_rows[i]["group"]
        for batch in order[:selected_step]
        for i in batch
        if train_rows[i]["source"] == "clevr4"
    }
    held = [
        row
        for row in train_rows
        if row["source"] == "clevr4" and row["group"] not in seen_groups
    ]
    if not held or any(row["id"] in seen_ids for row in held):
        raise ValueError("holdout contains a training-seen question or is empty")
    group_observations = {}
    observation_groups = {}
    for row in train_rows:
        if row["source"] != "clevr4":
            continue
        old = group_observations.setdefault(row["group"], row["observation"])
        if old != row["observation"]:
            raise ValueError("one image group maps to multiple observations")
        prior = observation_groups.setdefault(row["observation"], row["group"])
        if prior != row["group"]:
            raise ValueError("one image observation maps to multiple groups")
    if seen_groups & {row["group"] for row in held}:
        raise ValueError("image asset identity overlaps selected-checkpoint training")
    return held, seen_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha(args.reference / "result.json") != args.reference_sha256:
        raise ValueError("reference receipt hash differs")
    reference = load_json(args.reference / "result.json")
    config = load_json(CONFIG)
    corpus, queries = verify_inputs(config, reference)
    if reference.get("variant") != "baseline" or reference.get("best_step") != 1792:
        raise ValueError("pinned Path A baseline selected checkpoint required")

    for name, digest in reference["artifacts"].items():
        verify_file(args.reference / name, digest)
    train_rows = [
        json.loads(line)
        for line in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("only original train rows are admitted")
    sampling = load_json(args.reference / "sampling.json")
    holdout, seen_groups = unused_image_rows(train_rows, sampling["order"], 1792)
    if len(seen_groups) != 1672 or len({row["group"] for row in holdout}) != 6752:
        raise ValueError("unexpected selected-step image group accounting")
    if len(holdout) != 27008:
        raise ValueError("unexpected four-question-per-image holdout size")

    observations = load_json(corpus / "observations.json")
    banks = load_json(corpus / "banks.json")
    query_receipt = load_json(queries / "result.json")
    banks["new/queries.npy"] = dict(
        path=str(queries / "queries.npy"),
        sha256=query_receipt["artifacts"]["queries.npy"],
        shape=[2622, 384],
    )
    query_map = load_json(corpus / "query-map.json")

    checkpoint_path = args.reference / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("step") != 1792:
        raise ValueError("best checkpoint step differs")
    model = NativeCandidateReadout()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cpu()
    store = FeatureStore(observations, banks, query_map, device="cpu")
    args.output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    last_tick, last_count, samples = started, 0, []

    def progress(completed, total):
        nonlocal last_tick, last_count
        now = time.perf_counter()
        interval = now - last_tick
        rate = (completed - last_count) / interval if interval else math.inf
        samples.append(dict(rows=completed, interval_seconds=interval, rows_per_second=rate))
        average = float(np.mean([row["rows_per_second"] for row in samples[-3:]]))
        remaining = (total - completed) / average if average else math.inf
        print(
            f"image transfer {completed}/{total}; sample {rate:.1f} rows/s; "
            f"last-three mean {average:.1f}; ETA {remaining / 60:.1f} min",
            flush=True,
        )
        last_tick, last_count = now, completed

    scores = score_validation(model, holdout, store, progress=progress)
    elapsed = time.perf_counter() - started
    score_path = args.output / "scores.npy"
    np.save(score_path, scores, allow_pickle=False)
    metrics = summarize(holdout, scores)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ids = [dict(id=row["id"], group=row["group"], observation=row["observation"],
                target_index=row["target_index"]) for row in holdout]
    ids_path = args.output / "holdout-ids.json"
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    report = dict(
        status="PASS_CPU_UNUSED_TRAIN_IMAGE_TRANSFER",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        training_source_commit=reference["source_commit"],
        reference_receipt_sha256=args.reference_sha256,
        checkpoint_sha256=sha(checkpoint_path),
        selected_step=1792,
        selected_checkpoint_seen_image_groups=len(seen_groups),
        heldout_image_groups=len({row["group"] for row in holdout}),
        heldout_question_count=len(holdout),
        source_split="original train partition; never seen by selected checkpoint at step 1792",
        ids_sha256=sha(ids_path),
        score_sha256=sha(score_path),
        metrics_sha256=sha(metrics_path),
        cpu_only=True,
        cuda_used=False,
        training_performed=False,
        checkpoint_selection_performed=False,
        sealed_audit_loaded=False,
        is_blind_or_independent_final_evaluation=False,
        elapsed_seconds=elapsed,
        rows_per_second=len(holdout) / elapsed,
        first_three_rate_mean=float(np.mean([row["rows_per_second"] for row in samples[:3]])),
        throughput_samples=samples,
        macro=metrics["macro"],
        modalities=metrics["modalities"],
        note=(
            "This is a one-time, image-group-disjoint transfer check on records from the original "
            "training partition. It is not sealed audit evidence and does not support tuning."
        ),
    )
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
