"""Verify both frozen fits, transfer/replay and publish the complete matched comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.binding_language import quality_failures
from scripts.binding_transfer import control_summary
from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import retention_ok, select_checkpoint
from scripts.condition_binding_training import score_summary
from scripts.condition_contrast import CONDITIONS, eligible
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.prepare_condition_contrast import CONFIG, validate_corpus
from scripts.semantic_state_diagnostic import log_probabilities
from scripts.summarize_condition_binding import score_difference


def error_attribution(rows, prediction_ids):
    """Post-hoc saved-prediction description; never used for model selection."""
    result = dict(
        questions=len(rows),
        correct=0,
        wrong_field=0,
        correct_field_wrong_value=0,
        wrong_field_value_matches_state=0,
        target_to_predicted_field={},
    )
    for row, predicted in zip(rows, prediction_ids, strict=True):
        if predicted not in row["option_ids"]:
            raise ValueError("prediction must identify a supplied candidate")
        field, value = predicted.split(":", 1)
        target_field = row["target_id"].split(":", 1)[0]
        confusion = result["target_to_predicted_field"].setdefault(target_field, {})
        confusion[field] = confusion.get(field, 0) + 1
        if predicted == row["target_id"]:
            result["correct"] += 1
        elif field == target_field:
            result["correct_field_wrong_value"] += 1
        else:
            result["wrong_field"] += 1
            if value == str(json.loads(row["state"])[field]):
                result["wrong_field_value_matches_state"] += 1
    return result


def mechanism_failures(models, intervals, final_retained):
    failures = [] if final_retained else ["final_retention"]
    for name in ("focus_descriptions", "complement_descriptions"):
        a, b = models["control"][name]["overall"], models["candidate"][name]["overall"]
        checks = dict(
            improvement=b["accuracy"] - a["accuracy"] >= 0.03,
            paired_lower=intervals[name]["lower"][0] > 0,
            nll=b["nll"] < a["nll"],
            brier=b["brier"] <= a["brier"],
            ece=b["ece"] <= a["ece"] + 0.02,
        )
        failures.extend(f"{name}:{k}" for k, v in checks.items() if not v)
    return failures


def verify_curve(history, best_step):
    if [r["step"] for r in history] != [0, 128, 256, 384, 512] or select_checkpoint(
        history
    ) != best_step:
        raise ValueError("full fixed curve and validation-only selection required")
    for row in history:
        s = row["step"]
        expected = {} if s == 0 else dict(binding=4 * s, original=2 * s, curriculum=2 * s)
        if (
            row["consumed"] != 8 * s
            or row["unique_questions"] != 8 * s
            or row["source_counts"] != expected
        ):
            raise ValueError("consumption mismatch")


def load_run(root, name, status):
    path = root / name
    run = read(path / "result.json")
    if run["status"] != status:
        raise ValueError(f"incomplete run {name}")
    verify({str(path / p): h for p, h in run["artifacts"].items()})
    return run


def verify_run_provenance(run, input_pins, code_pins):
    if run["input_pins"] != input_pins or run["code_pins"] != code_pins:
        raise ValueError("run provenance does not match the current frozen experiment")
    verify(input_pins)
    verify(code_pins)


def fit_provenance(directory, fit):
    return {
        str(directory / "result.json"): sha(directory / "result.json"),
        **{str(directory / p): h for p, h in fit["artifacts"].items()},
    }


def build_publication(root):
    corpus = root / "condition-contrast-corpus-01"
    config = read(CONFIG)
    cp = read(corpus / "result.json")
    post_pins, code_pins = validate_corpus(corpus, sha(corpus / "result.json"), config)
    training_pins, training_code = validate_corpus(
        corpus, sha(corpus / "result.json"), config, training_only=True
    )
    if training_code != code_pins:
        raise ValueError("training/evaluation code provenance differs")
    runs = {}
    fits = {}
    curves = {}
    old = root / "condition-binding-corpus-01"
    data = {
        name: lines(corpus / "validation.jsonl")
        if name == "binding"
        else lines(old / f"{file}.jsonl")
        for name, file in [
            ("binding", "validation"),
            ("legacy_binding", "validation"),
            ("original", "original"),
            ("curriculum", "curriculum"),
        ]
    }
    fixed_train = lines(corpus / "train.jsonl")[:256]
    for arm, prefix in [("ce", "condition-contrast-ce"), ("contrast", "condition-contrast")]:
        fit_name = prefix + "-fit-01"
        replay_name = prefix + "-replay-01"
        fit = load_run(root, fit_name, "PASS_FIXED_CONDITION_CONTRAST")
        replay = load_run(root, replay_name, "PASS_SELECTED_CONDITION_CONTRAST")
        verify_run_provenance(fit, training_pins, code_pins)
        verify_run_provenance(replay, training_pins, code_pins)
        post_pins.update(fit_provenance(root / fit_name, fit))
        runs[fit_name], runs[replay_name] = fit, replay
        if (
            fit["arm"] != arm
            or replay["arm"] != arm
            or fit["input_pins"] != replay["input_pins"]
            or fit["code_pins"] != replay["code_pins"]
            or fit["selected_sha256"] != replay["selected_sha256"]
        ):
            raise ValueError("fit/replay mismatch")
        history = read(root / fit_name / "history.json")
        verify_curve(history, fit["best_step"])
        updates = read(root / fit_name / "updates.json")
        if [u["step"] for u in updates] != list(range(1, 513)) or any(
            u["contrast_coefficient"] != config["arms"][arm] for u in updates
        ):
            raise ValueError("update accounting or intervention mismatch")
        for row in history:
            s = row["step"]
            summaries = {
                n: score_summary(
                    rows,
                    np.load(root / fit_name / f"validation-{s}-{n}-scores.npy", allow_pickle=False),
                )
                for n, rows in data.items()
            }
            if (
                summaries != row["validation"]
                or score_summary(
                    fixed_train,
                    np.load(root / fit_name / f"fixed-train-{s}-scores.npy", allow_pickle=False),
                )
                != row["fixed_train"]
            ):
                raise ValueError("curve raw-score mismatch")
            base = history[0]["validation"]
            old_ok = retention_ok(
                {n: summaries[n]["overall"] for n in ("original", "curriculum")},
                {
                    n + ":" + t: m["accuracy"]
                    for n in ("original", "curriculum")
                    for t, m in summaries[n]["tasks"].items()
                },
                {n: base[n]["overall"] for n in ("original", "curriculum")},
                {
                    n + ":" + t: m["accuracy"]
                    for n in ("original", "curriculum")
                    for t, m in base[n]["tasks"].items()
                },
            )
            if row["eligible"] != eligible(
                s, old_ok, summaries["legacy_binding"]["overall"]["accuracy"]
            ):
                raise ValueError("retention gate mismatch")
        expected = [
            f"reload-{n}-{suffix}"
            for n in (*data, "fixed_train")
            for suffix in ("scores.npy", "probabilities.npy", "metrics.json")
        ]
        if set(replay["fresh_replay_exact"]) != set(expected) or any(
            replay["artifacts"][p] != fit["artifacts"][p] for p in expected
        ):
            raise ValueError("incomplete selected replay")
        fits[arm] = fit
        curves[arm] = history
    for n in data:
        if not np.array_equal(
            np.load(root / "condition-contrast-ce-fit-01" / f"validation-0-{n}-scores.npy"),
            np.load(root / "condition-contrast-fit-01" / f"validation-0-{n}-scores.npy"),
        ):
            raise ValueError("matched initialization differs")
    evaluation = "condition-contrast-transfer-01"
    rerun = "condition-contrast-transfer-replay-01"
    for name in (evaluation, rerun):
        runs[name] = load_run(root, name, "PASS_FIXED_CONDITION_CONTRAST_TRANSFER")
        verify_run_provenance(runs[name], post_pins, code_pins)
    post = read(root / evaluation / "metrics.json")
    arrays = {}
    attribution = {}
    transfer = {n: lines(corpus / f"{n}.jsonl") for n in CONDITIONS}
    for label, conditions in post["models"].items():
        arrays[label] = {}
        attribution[label] = {}
        for name, metric in conditions.items():
            rows = transfer[name]
            scores = np.load(root / evaluation / f"{label}-{name}-scores.npy", allow_pickle=False)
            arrays[label][name] = scores
            if scores.shape != (1024, 40) or score_summary(rows, scores) != metric:
                raise ValueError("transfer metric mismatch")
            probabilities = np.exp(log_probabilities(scores))
            saved = np.load(
                root / evaluation / f"{label}-{name}-probabilities.npy", allow_pickle=False
            )
            if not np.array_equal(probabilities, saved):
                raise ValueError("probability mismatch")
            predictions = lines(root / evaluation / f"{label}-{name}-predictions.jsonl")
            for row, p, item in zip(rows, probabilities, predictions, strict=True):
                if (
                    item["id"] != row["id"]
                    or item["group"] != row["group"]
                    or item["target_id"] != row["target_id"]
                    or item["option_ids"] != row["option_ids"]
                    or item["probabilities"] != p.tolist()
                    or item["prediction_id"] != row["option_ids"][int(p.argmax())]
                ):
                    raise ValueError("per-example prediction mismatch")
            attribution[label][name] = error_attribution(
                rows, [item["prediction_id"] for item in predictions]
            )
        controls = control_summary(
            arrays[label]["focus_descriptions"],
            arrays[label]["reverse_focus_descriptions"],
            arrays[label]["no_question"],
        )
        if controls != post["controls"][label]:
            raise ValueError("control mismatch")
    intervals = {
        n: score_difference(transfer[n], arrays["control"][n], arrays["candidate"][n])
        for n in CONDITIONS
    }
    for n, interval in intervals.items():
        for k in ("difference", "lower", "upper", "groups", "questions"):
            if interval[k] != post["paired_intervals"][n][k]:
                raise ValueError("interval mismatch")
    payloads = [
        p
        for p in runs[evaluation]["artifacts"]
        if p.endswith(".npy") or p.endswith("-predictions.jsonl")
    ] + ["metrics.json", "effective-config.json"]
    if set(runs[rerun]["fresh_replay_files_exact"]) != set(payloads) or any(
        runs[rerun]["artifacts"][p] != runs[evaluation]["artifacts"][p] for p in payloads
    ):
        raise ValueError("full transfer replay mismatch")
    final_retention = curves["contrast"][-1]["eligible"]
    failures = mechanism_failures(post["models"], intervals, final_retention)
    quality = quality_failures(post["models"]["candidate"], post["models"]["control"], intervals)
    if quality != post["transfer_gate_failures"]:
        raise ValueError("quality-gate mismatch")
    smoke = "condition-contrast-smoke-01"
    runs[smoke] = load_run(root, smoke, "PASS_TRAIN_ONLY_CONDITION_CONTRAST_SMOKE")
    verify_run_provenance(runs[smoke], training_pins, code_pins)
    return dict(
        status="PASS_MATCHED_CONDITIONAL_CONTRAST_EXECUTION",
        product_complete=False,
        selected_for_product=False,
        sealed_audit_loaded=False,
        source_commit=fits["ce"]["source_commit"],
        publication_code_pins={
            str(Path(p).resolve()): sha(p)
            for p in (
                "scripts/publish_condition_contrast.py",
                "tests/test_condition_contrast_publication.py",
            )
        },
        config=config,
        corpus_sha256=sha(corpus / "result.json"),
        corpus_summary={
            k: cp[k]
            for k in (
                "excluded_unique_states",
                "partitions",
                "rows",
                "train_families",
                "native_media_assets",
                "state_overlap",
            )
        },
        corpus_artifacts=cp["artifacts"],
        curves=curves,
        post_selection=post,
        post_hoc_error_attribution=attribution,
        mechanism_support=not failures,
        mechanism_failures=failures,
        final_quality_failures=quality,
        final_retention={arm: h[-1]["eligible"] for arm, h in curves.items()},
        selected_steps={arm: f["best_step"] for arm, f in fits.items()},
        selected_consumption={
            arm: next(h for h in curves[arm] if h["step"] == f["best_step"])["consumed"]
            for arm, f in fits.items()
        },
        selected_weight_hashes={arm: f["selected_sha256"] for arm, f in fits.items()},
        resources={
            n: {
                k: r.get(k)
                for k in (
                    "seconds",
                    "peak_vram_bytes",
                    "peak_host_bytes",
                    "candidate_pair_forward_presentations",
                    "actual_candidate_pairs",
                    "max_tokens",
                    "truncated_pair_presentations",
                    "started_utc",
                    "finished_utc",
                )
            }
            for n, r in runs.items()
        },
        run_manifest_pins={
            str(root / n / "result.json"): sha(root / n / "result.json") for n in runs
        },
        cloud_cost=0,
        exact_selected_replay_files_per_arm=15,
        exact_post_replay_files=len(payloads),
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
