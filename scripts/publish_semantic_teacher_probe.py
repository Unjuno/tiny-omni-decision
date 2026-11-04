"""Verify saved teacher screen outputs and publish a compact, task-scoped record."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.semantic_teacher_probe import probabilities, summarize_rows


def verify_replay_cohort(rows, replay):
    first = {
        source: next(r["group"] for r in rows if r["source"] == source)
        for source in ("binding", "numeric")
    }
    expected = [r["id"] for r in rows if r["group"] == first[r["source"]]]
    actual = [p["row_id"] for p in replay]
    if len(expected) != 36 or actual != expected or len(set(actual)) != len(actual):
        raise ValueError("exact planned 36-row replay cohort required")


def paired_reversal(rows, predictions, source):
    original, reversed_name = (
        ("exclusion", "reverse_exclusion") if source == "binding" else ("original", "reverse")
    )
    groups = defaultdict(dict)
    for row, prediction in zip(rows, predictions, strict=True):
        if row["source"] == source and row["condition"] in (original, reversed_name):
            key = row["task"] if source == "binding" else row["id"].split(":")[-2]
            groups[row["group"]].setdefault(key, {})[row["condition"]] = float(
                prediction["prediction_id"] == row["target_id"]
            )
    differences = np.array(
        [
            np.mean([v[reversed_name] - v[original] for v in block.values()])
            for _, block in sorted(groups.items())
        ]
    )
    rng = np.random.default_rng(17)
    samples = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(1)
    return dict(
        unit="TRAIN source-qualified state",
        groups=len(differences),
        draws=10000,
        seed=17,
        estimand="reversed minus original Accuracy",
        point=float(differences.mean()),
        percentile95=np.quantile(samples, [0.025, 0.975]).tolist(),
        blind_evaluation=False,
    )


def build_publication(root):
    names = [
        "semantic-teacher-corpus-04",
        "semantic-teacher-smoke-01",
        "semantic-teacher-full-01",
        "semantic-teacher-replay-01",
    ]
    runs = {name: read(root / name / "result.json") for name in names}
    expected = [
        "PASS_PREPARED_TRAIN_ONLY",
        "PASS_SMOKE",
        "PASS_FULL_TRAIN_SCREEN",
        "PASS_BOUNDED_REPLAY",
    ]
    for name, status in zip(names, expected, strict=True):
        run = runs[name]
        if run["status"] != status or not run["old_inputs_unchanged"]:
            raise ValueError("incomplete or unverified run")
        verify({str(root / name / file): digest for file, digest in run["artifacts"].items()})
        if (
            run["config"] != runs[names[0]]["config"]
            or run["code_pins"] != runs[names[0]]["code_pins"]
        ):
            raise ValueError("run protocol drift")
    rows = read(root / names[0] / "rows.json")
    predictions = read(root / names[2] / "predictions.json")
    full = runs[names[2]]
    if summarize_rows(rows, predictions) != full["metrics"]:
        raise ValueError("derived metric mismatch")
    for row, prediction in zip(rows, predictions, strict=True):
        p = probabilities(prediction["logits"])
        if (
            prediction["probabilities"] != p.tolist()
            or prediction["target_id"] != row["target_id"]
            or prediction["prediction_id"] != row["option_ids"][int(p.argmax())]
        ):
            raise ValueError("saved probabilities or identities mismatch")
    lookup = {p["row_id"]: p for p in predictions}
    replay = read(root / names[3] / "predictions.json")
    verify_replay_cohort(rows, replay)
    if any(p != lookup[p["row_id"]] for p in replay):
        raise ValueError("bounded replay payload mismatch")
    controls = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        if row["condition"] == "no_question":
            controls[row["group"]].append(prediction["logits"])
    if any(len(v) != 4 or any(a != v[0] for a in v) for v in controls.values()):
        raise ValueError("empty-question predictions depend on hidden task metadata")
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
    screen_pass = all(full["reversal_pass"].values()) and all(
        all(tasks.values()) for tasks in full["train_screen_eligible_by_task"].values()
    )
    return dict(
        status="PASS_EXECUTION_SCREEN_ONLY"
        if screen_pass
        else "PASS_EXECUTION_TEACHER_NOT_ADMITTED",
        teacher_screen_pass=screen_pass,
        product_complete=False,
        student_training_started=False,
        source_commit=full["source_commit"],
        config=full["config"],
        input_pins=full["input_pins"],
        code_pins=full["code_pins"],
        model_pins=full["model_pins"],
        environment=full["environment"],
        metrics=full["metrics"],
        allowed_label_mass=mass,
        train_screen_eligible_by_task=full["train_screen_eligible_by_task"],
        reversal_pass=full["reversal_pass"],
        paired_reversal={s: paired_reversal(rows, predictions, s) for s in ("binding", "numeric")},
        replay_rows=len(replay),
        replay_exact=True,
        no_question_equal_quartets=len(controls),
        total_parameters=full["total_parameters"],
        trainable_parameters=0,
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
