"""Freeze all development input replacements before model scoring; metadata only."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts.greatest_hits_cache import frozen_events
from scripts.greatest_hits_matched_inputs import TASKS, Pins, read
from scripts.greatest_hits_replacement import build_inventory
from scripts.native_video_cache import sha
from scripts.probe_greatest_hits_transport import write

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs/pretrained_reuse/R2_GREATEST_HITS_INPUT_REPLACEMENT_PROTOCOL_20261017.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip()
    ):
        raise ValueError("clean committed source and CUDA hidden required")
    args.output.mkdir(exist_ok=False)
    started, pins = time.perf_counter(), Pins()
    result = dict(
        status="RUNNING",
        command=[sys.executable, *sys.argv],
        pid=os.getpid(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        started_utc=datetime.now(UTC).isoformat(),
        optimizer_updates=0,
        encoder_updates=0,
        model_forwards=0,
        quality_evaluated=False,
        features_loaded=False,
        validation_metadata_loaded=True,
        sealed_audit_loaded=False,
        calibration_loaded=False,
        cloud_cost=0,
        checkpoint_selection=False,
    )
    try:
        pins.add(
            TASKS / "result.json",
            "7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831",
        )
        receipt = read(TASKS / "result.json")
        if receipt["status"] != "PASS_FIXED_DEVELOPMENT_TASKS":
            raise ValueError("original passing development tasks required")
        pins.add(PROTOCOL)
        for name in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0"):
            if name:
                pins.add(REPO / name)
        splits = {}
        for split in ("train", "validation"):
            path = TASKS / (split + ".jsonl")
            pins.add(path, receipt["artifacts"][path.name])
            splits[split] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
        events = frozen_events(splits)
        if len(events) != 11151 or len(splits["validation"]) != 8875:
            raise ValueError("complete frozen development metadata required")
        rows = splits["validation"]
        plan = build_inventory(rows)
        if build_inventory(rows[::-1]) != plan:
            raise ValueError("input row order changed donor choices")
        permuted_labels = [dict(r, target=(r["target"] + 1) % len(r["options"])) for r in rows]
        if build_inventory(permuted_labels) != plan:
            raise ValueError("labels changed donor choices")
        by_id = {r["id"]: r for r in rows}
        agreement = Counter()
        for pair in plan["pairs"]:
            original, donor = by_id[pair["id"]], by_id[pair["donor_id"]]
            agreement[pair["task"], original["target"] == donor["target"]] += 1
        write(args.output / "inventory.json", plan)
        write(
            args.output / "source-label-agreement.json",
            [dict(task=k[0], equal=k[1], count=v) for k, v in sorted(agreement.items())],
        )
        pins.verify()
        result.update(
            status="PASS_FROZEN_AV_REPLACEMENT_INVENTORY",
            questions=len(plan["pairs"]),
            views_per_question=len(plan["views"]),
            planned_forwards_per_arm=len(plan["pairs"]) * len(plan["views"]),
            pairs_sha256=plan["pairs_sha256"],
            unique_donors=plan["unique_donor_questions"],
            max_donor_uses=plan["max_donor_uses"],
            source_label_agreement_is_counterfactual_truth=False,
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
    except BaseException as exc:
        result.update(status="INCOMPLETE", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - started,
            finished_utc=datetime.now(UTC).isoformat(),
            pins=pins.hashes,
        )
        write(args.output / "result.json", result)
        print(result["status"], flush=True)


if __name__ == "__main__":
    main()
