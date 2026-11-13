"""CPU-only reconstruction of the fixed replay source coverage comparison."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.binding_language import CONDITIONS, quality_failures
from scripts.binding_transfer import control_summary
from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import select_checkpoint
from scripts.condition_binding_training import score_summary
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.publish_binding_language import check_run_modes, cross_run_checks
from scripts.summarize_condition_binding import gate_failures, score_difference


def check_control_arrays(old, current):
    if set(old) != set(current) or any(not np.array_equal(old[k], current[k]) for k in old):
        raise ValueError("unchanged control regression scores differ")


def check_accounting(history, data, order):
    for row in history:
        used = [data[source][index] for batch in order[: row["step"]] for source, index in batch]
        expected = dict(
            consumed=len(used),
            unique_questions=len({r["id"] for r in used}),
            source_counts=dict(Counter(r["source"] for r in used)),
            task_counts=dict(Counter(r["source"] + ":" + r["task"] for r in used)),
            language_family_counts=dict(
                Counter(str(r["language_family"]) for r in used if r["source"] == "binding")
            ),
            unique_states_by_source={
                s: len({r["group"] for r in used if r["source"] == s})
                for s in ("binding", "original", "curriculum")
            },
        )
        if len(used) != expected["unique_questions"] or any(
            row[k] != v for k, v in expected.items()
        ):
            raise ValueError("recorded consumption differs from frozen sampler")


def hypothesis(eligible, curriculum_difference):
    lower_nll = curriculum_difference["difference"][1] < 0
    if eligible and lower_nll and curriculum_difference["upper"][1] < 0:
        return "SUPPORT_WITHIN_FIXED_BUDGET"
    if not eligible and lower_nll:
        return "MIXED_RETENTION_TRADEOFF"
    if eligible or lower_nll:
        return "UNCERTAIN"
    return "NOT_SUPPORTED_WITHIN_FIXED_BUDGET"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for key in ("smoke", "fit", "selected_replay", "transfer", "transfer_replay"):
        parser.add_argument("--" + key.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("publication output already exists")
    specifications = {
        "smoke": ("binding-replay-coverage-smoke-01", "PASS_TRAIN_ONLY_REPLAY_COVERAGE_SMOKE"),
        "fit": ("binding-replay-coverage-fit-01", "PASS_FIXED_REPLAY_COVERAGE"),
        "selected_replay": (
            "binding-replay-coverage-selected-replay-01",
            "PASS_SELECTED_REPLAY_COVERAGE",
        ),
        "transfer": (
            "binding-replay-coverage-regression-01",
            "PASS_FIXED_REPLAY_COVERAGE_REGRESSION",
        ),
        "transfer_replay": (
            "binding-replay-coverage-regression-replay-01",
            "PASS_FIXED_REPLAY_COVERAGE_REGRESSION",
        ),
    }
    results, result_pins = {}, {}
    for key, (directory, status) in specifications.items():
        path = args.root / directory
        digest = getattr(args, key + "_sha256")
        result_pins[str(path / "result.json")] = digest
        verify({str(path / "result.json"): digest})
        value = read(path / "result.json")
        if value["status"] != status or not value["sources_unchanged"]:
            raise ValueError(f"incomplete or changed run: {key}")
        verify({str(path / p): h for p, h in value["artifacts"].items()})
        verify(value["input_pins"])
        verify(value["code_pins"])
        results[key] = value
    check_run_modes(results)
    cross_run_checks(
        results, str(args.root / "binding-replay-coverage-fit-01/result.json"), args.fit_sha256
    )
    fit = results["fit"]
    fit_dir = args.root / specifications["fit"][0]
    history = read(fit_dir / "history.json")
    updates = read(fit_dir / "updates.json")
    if (
        [r["step"] for r in history] != [0, 128, 256, 384, 512]
        or [r["step"] for r in updates] != list(range(1, 513))
        or fit["steps"] != 512
        or select_checkpoint(history) != fit["best_step"]
    ):
        raise ValueError("incomplete training or selector mismatch")
    if (
        results["selected_replay"]["selected_sha256"] != fit["selected_sha256"]
        or len(results["selected_replay"]["fresh_replay_exact"]) != 12
    ):
        raise ValueError("selected replay incomplete")
    effective = read(fit_dir / "effective-config.json")
    old_corpus = args.root / "condition-binding-corpus-01"
    new_corpus = args.root / "binding-language-corpus-01"
    data = {
        "binding": lines(old_corpus / "validation.jsonl"),
        "original": lines(old_corpus / "original.jsonl"),
        "curriculum": lines(old_corpus / "curriculum.jsonl"),
        "fixed_train": lines(new_corpus / "train.jsonl")[:256],
    }
    replay_corpus = args.root / "binding-replay-coverage-corpus-01"
    check_accounting(
        history,
        [lines(replay_corpus / "train.jsonl"), lines(replay_corpus / "replay.jsonl")],
        read(replay_corpus / "sample-order.json"),
    )
    curve = []
    for item in history:
        for source, rows in data.items():
            filename = (
                f"fixed-train-{item['step']}-scores.npy"
                if source == "fixed_train"
                else f"validation-{item['step']}-{source}-scores.npy"
            )
            metrics = score_summary(rows, np.load(fit_dir / filename, allow_pickle=False))
            expected = (
                item["fixed_train"] if source == "fixed_train" else item["validation"][source]
            )
            if metrics != expected:
                raise ValueError(f"learning curve score mismatch: {source}")
        failures = gate_failures(item, history[0])
        if bool(failures) == item["eligible"]:
            raise ValueError("retention eligibility mismatch")
        curve.append(dict(**item, retention_failures=failures))
    control_dir = args.root / "binding-language-fit-01"
    comparisons = {}
    for source in ("binding", "original", "curriculum"):
        comparisons[source] = score_difference(
            data[source],
            np.load(control_dir / f"validation-512-{source}-scores.npy"),
            np.load(fit_dir / f"validation-512-{source}-scores.npy"),
        )
        comparisons[source]["columns"] = [
            "accuracy_candidate512_minus_control512",
            "nll_candidate512_minus_control512",
        ]
    selected = next(r for r in curve if r["step"] == fit["best_step"])
    selected_vs_parent = {}
    for source in ("binding", "original", "curriculum"):
        selected_vs_parent[source] = score_difference(
            data[source],
            np.load(fit_dir / f"validation-0-{source}-scores.npy"),
            np.load(fit_dir / f"validation-{fit['best_step']}-{source}-scores.npy"),
        )
    transfer_dir = args.root / "binding-replay-coverage-regression-01"
    transfer = read(transfer_dir / "metrics.json")
    expected_models = {"control", "candidate"} | (
        {"selected"} if fit["best_step"] != 512 else set()
    )
    if (
        set(transfer["models"]) != expected_models
        or transfer["selected_step"] != fit["best_step"]
        or transfer["selected_model"] != ("candidate" if fit["best_step"] == 512 else "selected")
    ):
        raise ValueError("transfer checkpoint labels mismatch")
    arrays = {}
    for model, summaries in transfer["models"].items():
        if set(summaries) != set(CONDITIONS):
            raise ValueError("transfer conditions mismatch")
        arrays[model] = {}
        for condition, expected in summaries.items():
            rows = lines(new_corpus / f"{condition}.jsonl")
            scores = np.load(transfer_dir / f"{model}-{condition}-scores.npy")
            if score_summary(rows, scores) != expected:
                raise ValueError("heldout metrics mismatch")
            arrays[model][condition] = scores
        actual_control = control_summary(
            arrays[model]["focus_descriptions"],
            arrays[model]["reverse_focus_descriptions"],
            arrays[model]["no_question"],
        )
        no = summaries["no_question"]
        if (
            actual_control != transfer["controls"][model]
            or no["overall"]["accuracy"] > 0.25
            or no["pair_both_correct"] != 0
            or no["all_four_correct"] != 0
        ):
            raise ValueError("transfer control mismatch")
    old_regression = args.root / "binding-language-transfer-01"
    old_digest = "52b9b5ae11c80a69b4071d1fc4ed714adc9265ad1833d90a88faa4fbe5c3faa2"
    verify({str(old_regression / "result.json"): old_digest})
    old_result = read(old_regression / "result.json")
    verify({str(old_regression / p): h for p, h in old_result["artifacts"].items()})
    check_control_arrays(
        {n: np.load(old_regression / f"candidate-{n}-scores.npy") for n in CONDITIONS},
        arrays["control"],
    )
    result_pins[str(old_regression / "result.json")] = old_digest
    for condition in CONDITIONS:
        rows = lines(new_corpus / f"{condition}.jsonl")
        interval = score_difference(
            rows, arrays["control"][condition], arrays["candidate"][condition]
        )
        interval["columns"] = [
            "accuracy_candidate512_minus_control512",
            "nll_candidate512_minus_control512",
        ]
        if interval != transfer["paired_intervals"][condition]:
            raise ValueError("paired interval mismatch")
    if transfer["transfer_gate_failures"] != quality_failures(
        transfer["models"]["candidate"], transfer["models"]["control"], transfer["paired_intervals"]
    ):
        raise ValueError("heldout quality rule mismatch")
    if len(results["transfer_replay"]["fresh_replay_files_exact"]) != (
        len(transfer["models"]) * 7 * 3 + 2
    ):
        raise ValueError("full transfer replay incomplete")
    selected_fields = (
        "step",
        "consumed",
        "unique_questions",
        "unique_states_by_source",
        "source_counts",
        "task_counts",
        "language_family_counts",
    )
    write(
        args.output,
        dict(
            status="PASS_EXECUTION_AND_CPU_RECONSTRUCTION",
            result_pins=result_pins,
            effective_config=effective,
            corpus=read(replay_corpus / "result.json"),
            language_regression_corpus=read(new_corpus / "result.json"),
            hypothesis=hypothesis(curve[-1]["eligible"], comparisons["curriculum"]),
            evaluation_scope="previously observed development regression; not blind audit",
            curve=curve,
            fixed512_vs_control=comparisons,
            selected_vs_parent=selected_vs_parent,
            selected_accounting={k: selected[k] for k in selected_fields},
            full_accounting={k: curve[-1][k] for k in selected_fields},
            transfer=transfer,
            control_history=read(control_dir / "history.json"),
            resources={
                k: {
                    f: r[f]
                    for f in (
                        "seconds",
                        "peak_vram_bytes",
                        "peak_host_bytes",
                        "candidate_pair_forward_presentations"
                        if k in ("fit", "smoke", "selected_replay")
                        else "actual_candidate_pairs",
                    )
                }
                for k, r in results.items()
            },
            total_recorded_process_body_seconds=sum(r["seconds"] for r in results.values()),
            update_seconds=sum(r["seconds"] for r in updates),
            scheduled_evaluation_seconds=sum(r["evaluation_seconds"] for r in history),
            gradient_norm_min=min(r["gradient_norm"] for r in updates),
            gradient_norm_max=max(r["gradient_norm"] for r in updates),
            initial_retention_kl=updates[0]["retention_kl"],
            best_bytes=(fit_dir / "best-delta.pt").stat().st_size,
            final_bytes=(fit_dir / "final-delta.pt").stat().st_size,
            best_sha256=sha(fit_dir / "best-delta.pt"),
            final_sha256=sha(fit_dir / "final-delta.pt"),
            transfer_quality_pass=not transfer["transfer_gate_failures"],
            selected_retention_pass=selected["eligible"],
            final_retention_pass=curve[-1]["eligible"],
            publisher_sha256=sha(__file__),
            product_complete=False,
            existing_artifacts_unchanged=True,
            sealed_audit_loaded=False,
            cloud_cost=0,
            resume_supported=False,
        ),
    )
    print(str(args.output), sha(args.output))


if __name__ == "__main__":
    main()
