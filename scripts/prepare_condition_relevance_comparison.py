"""Freeze both completed fits and compare saved validation; never open transfer."""

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.condition_relevance_corpus import load_development
from scripts.condition_relevance_postfit_evidence import audit_fit_evidence
from scripts.condition_relevance_postfit_io import (
    _json,
    _no_redirect,
    _outside_protected,
    _verified_bytes,
    capture_execution_source,
    verify_pair_files,
    write_freeze,
)
from scripts.condition_relevance_training import bootstrap_relevance, metric_summary


def _write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(run_directories, result_hashes, source_root, output):
    checked = verify_pair_files(run_directories, result_hashes, source_root)
    execution = capture_execution_source(
        source_root, __file__, "prepare_condition_relevance_comparison"
    )
    audits, endpoints, selected = {}, {}, {}
    schema_path = checked["corpus_root"] / "selection-schema.json"
    schema = _json(schema_path, checked["runs"]["ce"]["input_pins"][str(schema_path)])
    config = checked["runs"]["ce"]["config"]
    rows = []
    for partition in ("validation", "old_validation", "legacy_validation"):
        rows.extend(load_development(checked["corpus_root"], partition, config["corpus_sha256"]))
    for arm, run in checked["runs"].items():
        directory = checked["run_directories"][arm]

        def read(name, directory=directory, run=run):
            return _json(directory / name, run["artifacts"][name])

        updates = [
            json.loads(line)
            for line in _verified_bytes(
                directory / "updates.jsonl", run["artifacts"]["updates.jsonl"]
            ).splitlines()
        ]
        audits[arm] = audit_fit_evidence(
            run, read("history.json"), updates, read("sample-order.json"), schema
        )

        def scores(step, directory=directory, run=run):
            name = f"validation-{step}-scores.npy"
            return np.load(
                io.BytesIO(_verified_bytes(directory / name, run["artifacts"][name])),
                allow_pickle=False,
            )

        endpoints[arm] = scores(1024)
        selected[arm] = metric_summary(rows, scores(run["best_step"]), schema)
    for key in ("lr_consumption_sha256", "sampler_sha256"):
        if audits["ce"][key] != audits["relevance"][key]:
            raise ValueError("matched pair timeline mismatch: " + key)
    metrics = {arm: metric_summary(rows, values, schema) for arm, values in endpoints.items()}
    new_indices = [i for i, row in enumerate(rows) if row["source"] == "condition_relevance_v1"]
    if len(new_indices) != 2048:
        raise ValueError("all2048 new validation questions required")
    bootstrap = bootstrap_relevance(
        [rows[i] for i in new_indices],
        endpoints["ce"][new_indices],
        endpoints["relevance"][new_indices],
        replicates=10000,
        seed=17,
    )
    promising = (
        bootstrap["difference"] >= 0.03
        and bootstrap["interval95"][0] > 0
        and audits["relevance"]["best_step"] > 0
    )
    comparison = dict(
        scope="fixed1024 endpoints on observed development validation; not blind audit",
        primary_endpoint_metrics=metrics,
        selected_checkpoint_metrics=selected,
        primary_minimum_slice_bootstrap=bootstrap,
        predefined_auxiliary_promising_gate=promising,
        product_adoption=False,
        criterion_changed=False,
        fit_evidence=audits,
        retained_rule=(
            "minimum new16 Accuracy >=+3pp with positive95% lower bound "
            "AND trained selected checkpoint retains all47 slices"
        ),
    )
    output = _outside_protected(checked, output)
    output.mkdir(parents=True, exist_ok=False)
    freeze_sha = write_freeze(checked, output / "comparison-freeze.json")
    comparison_sha = _write(output / "validation-comparison.json", comparison)
    receipt = dict(
        status="PASS_FROZEN_CONDITION_RELEVANCE_COMPARISON",
        finished_utc=datetime.now(UTC).isoformat(),
        source_root=str(_no_redirect(source_root)),
        training_source_commit=checked["frozen"]["source_commit"],
        run_directories={a: str(p) for a, p in checked["run_directories"].items()},
        result_hashes=result_hashes,
        artifacts={
            "comparison-freeze.json": freeze_sha,
            "validation-comparison.json": comparison_sha,
        },
        **execution,
        transfer_loaded=False,
        calibration_loaded=False,
        sealed_audit_loaded=False,
        model_loaded=False,
        optimizer_updates=0,
        product_complete=False,
    )
    _write(output / "result.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce-directory", required=True, type=Path)
    parser.add_argument("--ce-sha256", required=True)
    parser.add_argument("--relevance-directory", required=True, type=Path)
    parser.add_argument("--relevance-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(
        {"ce": args.ce_directory, "relevance": args.relevance_directory},
        {"ce": args.ce_sha256, "relevance": args.relevance_sha256},
        args.source_root,
        args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
