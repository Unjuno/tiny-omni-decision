"""Verify the fixed factorial diagnostic and publish all arms without model selection."""

import argparse
from pathlib import Path

import numpy as np

from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.semantic_teacher_probe import probabilities
from scripts.teacher_label_factorial import factorial_summary, summarize_factorial_rows


def verify_cohort(rows, replay):
    first = {s: next(r["group"] for r in rows if r["source"] == s) for s in ("binding", "numeric")}
    expected = [r["id"] for r in rows if r["group"] == first[r["source"]]]
    actual = [p["row_id"] for p in replay]
    if len(expected) != 48 or actual != expected or len(set(actual)) != len(actual):
        raise ValueError("exact planned 48-row cohort required")


def verify_endpoints(rows, predictions, parent, expected_count=768):
    differences = []
    for row, item in zip(rows, predictions, strict=True):
        if row["id"] != item["row_id"] or row["baseline_row_id"] != item["baseline_row_id"]:
            raise ValueError("endpoint identity mismatch")
        if row["baseline_row_id"] is None:
            continue
        old = parent[row["baseline_row_id"]]
        diff = float(np.max(np.abs(np.array(item["logits"]) - old["logits"])))
        if (
            not np.isfinite(diff)
            or diff > 1e-5
            or item["prediction_id"] != old["prediction_id"]
            or item["target_id"] != old["target_id"]
            or item["target_id"] != row["target_id"]
        ):
            raise ValueError("historical endpoint drift")
        differences.append(diff)
    if len(differences) != expected_count:
        raise ValueError("incomplete historical endpoint coverage")
    return max(differences)


def build_publication(root):
    names = [f"teacher-label-factorial-{s}-01" for s in ("corpus", "smoke", "full", "replay")]
    runs = {n: read(root / n / "result.json") for n in names}
    expected = [
        "PASS_PREPARED_TRAIN_ONLY",
        "PASS_SMOKE",
        "PASS_FULL_TRAIN_SCREEN",
        "PASS_BOUNDED_REPLAY",
    ]
    corpus, full = runs[names[0]], runs[names[2]]
    for name, status in zip(names, expected, strict=True):
        run = runs[name]
        if run["status"] != status or not run["old_inputs_unchanged"]:
            raise ValueError("incomplete run")
        for key in ("config", "code_pins", "model_pins"):
            if run[key] != corpus[key]:
                raise ValueError("protocol drift")
        verify({str(root / name / file): digest for file, digest in run["artifacts"].items()})
        verify(run["input_pins"])
    verify(full["code_pins"])
    # Model files were hashed before/after each run; verify the publication's live source too.
    verify(full["model_pins"])
    rows = read(root / names[0] / "rows.json")
    predictions = read(root / names[2] / "predictions.json")
    if (
        len(rows) != 1536
        or len({r["id"] for r in rows}) != 1536
        or len({(r["source"], r["group"]) for r in rows}) != 64
        or summarize_factorial_rows(rows, predictions) != full["metrics"]
        or factorial_summary(rows, predictions) != full["factorial"]
    ):
        raise ValueError("incomplete cohort or derived metrics mismatch")
    for row, item in zip(rows, predictions, strict=True):
        p = probabilities(item["logits"])
        if (
            item["probabilities"] != p.tolist()
            or item["target_id"] != row["target_id"]
            or item["prediction_id"] != row["option_ids"][int(p.argmax())]
            or item["arm"] != row["arm"]
        ):
            raise ValueError("prediction mismatch")
    lookup = {p["row_id"]: p for p in predictions}
    for index in (1, 3):
        bounded = read(root / names[index] / "predictions.json")
        verify_cohort(rows, bounded)
        if any(p != lookup[p["row_id"]] for p in bounded):
            raise ValueError("smoke/replay output mismatch")
    parent = {
        p["row_id"]: p for p in read(Path(full["config"]["parent_full"]) / "predictions.json")
    }
    diff = verify_endpoints(rows, predictions, parent)
    mass = {}
    for condition in full["metrics"]["conditions"]:
        values = [
            p["option_probability_mass"]
            for r, p in zip(rows, predictions, strict=True)
            if f"{r['source']}/{r['condition']}" == condition
        ]
        mass[condition] = dict(
            minimum=min(values), median=float(np.median(values)), mean=float(np.mean(values))
        )
    return dict(
        status="PASS_FIXED_MECHANISM_DIAGNOSTIC",
        teacher_adopted=False,
        product_complete=False,
        student_training_started=False,
        audit_loaded=False,
        source_commit=full["source_commit"],
        config=full["config"],
        code_pins=full["code_pins"],
        model_pins=full["model_pins"],
        input_pins=full["input_pins"],
        environment=full["environment"],
        metrics=full["metrics"],
        factorial=full["factorial"],
        allowed_label_mass=mass,
        total_parameters=full["total_parameters"],
        trainable_parameters=0,
        rows=len(rows),
        unique_states=64,
        historical_endpoint_rows=768,
        historical_endpoint_max_abs_logit_difference=diff,
        replay_rows=48,
        smoke_and_replay_exact=True,
        full_start_utc=full["started_utc"],
        full_end_utc=full["ended_utc"],
        resources={
            name: {
                k: run.get(k)
                for k in (
                    "elapsed_seconds",
                    "load_seconds",
                    "inference_seconds",
                    "peak_vram_bytes",
                    "peak_host_bytes",
                )
            }
            for name, run in runs.items()
        },
        run_manifest_pins={
            str(root / name / "result.json"): sha(root / name / "result.json") for name in names
        },
        cloud_cost=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build_publication(args.root)
    write(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
