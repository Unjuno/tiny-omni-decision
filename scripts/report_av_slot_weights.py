"""Publish a hash-bound fixed coefficient comparison, including rejected results."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.av_slot_weight_metrics import eligible
from scripts.lombard_correspondence import choice_metrics
from scripts.replay_native_av_correspondence import sha, verify

METRICS = ["accuracy", "nll", "brier", "ece", "mean_confidence"]
REPLAY_KEYS = [
    "weights",
    "query_temperature",
    "calibration",
    "training_accounting",
    "train_curves",
    "policies",
    "eligible_for_runtime_trial",
]


def merge_pins(pins, incoming):
    for name, digest in incoming.items():
        if name in pins and pins[name] != digest:
            raise ValueError("conflicting physical input hash: " + name)
        pins[name] = digest


def admit(primary, replay, smoke):
    for data in [primary, replay]:
        if (
            data["status"] != "PASS_FIXED_SLOT_WEIGHT_COMPARISON"
            or data["smoke"]
            or data["iterations"] != 512
            or data["fit_parameter_count"] != 6
            or data["base_model_updates"] != 0
            or data["gpu_used"]
            or data["encoders_loaded"]
            or data["sealed_audit_opened"]
            or not data["source_and_base_weights_unchanged"]
            or not data["weights_frozen_before_calibration"]
            or not data["readout_frozen_before_validation"]
        ):
            raise ValueError("complete fixed TRAIN-only comparison required")
    if primary["fresh_process_replay"] or not replay["fresh_process_replay"]:
        raise ValueError("primary and actual fresh replay required")
    if (
        smoke["status"] != "PASS_TRAIN_ONLY_SLOT_WEIGHT_SMOKE"
        or smoke["calibration_evaluated"]
        or smoke["validation_evaluated"]
    ):
        raise ValueError("TRAIN-only smoke required")


def decision(data):
    reports = {}
    for policy, report in data["policies"].items():
        candidate = report["calibrated"]["validation"]
        baseline = report["baseline"]
        admitted = eligible(candidate, baseline, report["strata"])
        if admitted != report["eligible_for_runtime_trial"]:
            raise ValueError("policy eligibility inconsistent")
        reports[policy] = dict(
            eligible_for_runtime_trial=admitted,
            train_minus_validation_accuracy_pp=100
            * (report["calibrated"]["train"]["accuracy"] - candidate["accuracy"]),
            candidate_minus_baseline={
                k: candidate[k] - baseline[k] for k in baseline if k in METRICS
            },
            nll_relative_improvement=(baseline["nll"] - candidate["nll"]) / baseline["nll"],
        )
    admitted = all(p["eligible_for_runtime_trial"] for p in reports.values())
    if admitted != data["eligible_for_runtime_trial"]:
        raise ValueError("overall eligibility inconsistent")
    return dict(
        eligible_for_runtime_trial=admitted,
        action="ELIGIBLE_FOR_SEPARATE_RUNTIME_TRIAL_ONLY"
        if admitted
        else "CLOSE_SIX_COEFFICIENT_COMPARISON_KEEP_PARENT",
        policies=reports,
        product_promotion=False,
        four_modality_goal_met=False,
    )


def table_metrics(lines, policy, report):
    lines += [
        f"## {policy} candidates",
        "",
        "| Variant / split | Accuracy | NLL | Brier | ECE | Mean confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    values = [("Parent / development", report["baseline"])]
    for mode in ["raw", "calibrated"]:
        values += [(f"Candidate {mode} / {split}", m) for split, m in report[mode].items()]
    for name, metric in values:
        lines.append("| " + name + " | " + " | ".join(f"{metric[k]:.8f}" for k in METRICS) + " |")
    lines += [
        "",
        "Paired candidate-minus-parent intervals: whole development speaker groups,",
        "10,000 resamples, seed17, linear percentile95%. Nine groups; quartet roles",
        "are correlated. This observed development comparison is not a blind test.",
        "",
        "| Metric | Difference | Lower | Upper |",
        "|---|---:|---:|---:|",
    ]
    for key, interval in report["candidate_minus_baseline"]["differences"].items():
        lines.append(
            f"| {key} | {interval['point']:.8f} | "
            f"{interval['lower']:.8f} | {interval['upper']:.8f} |"
        )
    lines += [
        "",
        "| Stratum | Roles | Parent Acc | Candidate Acc | Candidate NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in report["strata"]:
        lines.append(
            f"| {s['category']}={s['value']} | {s['count']} | "
            f"{s['baseline']['accuracy']:.8f} | "
            f"{s['candidate']['accuracy']:.8f} | {s['candidate']['nll']:.8f} |"
        )
    lines += [
        "",
        "| Control | Original Accuracy | Original NLL | Corrected Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, c in report["controls"].items():
        corrected = f"{c['corrected']['accuracy']:.8f}" if "corrected" in c else "n/a"
        lines.append(
            f"| {name} | {c['original']['accuracy']:.8f} | "
            f"{c['original']['nll']:.8f} | {corrected} |"
        )
    lines += [
        "",
        "Absent-input uniform25% is the explicit prototype policy, not a trained",
        "unimodal result. Swap corrected labels restore matched metrics. All3556",
        "custom-ID reverse-order decisions are exact for this policy.",
        "",
    ]


def main():
    parser = argparse.ArgumentParser()
    for name in ["primary", "replay", "smoke"]:
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    pins = {}
    receipts = {}
    folders = {}
    for name in ["primary", "replay", "smoke"]:
        folder = getattr(args, name)
        receipt = folder / "result.json"
        digest = getattr(args, name + "_sha256")
        verify({str(receipt): digest})
        data = json.loads(receipt.read_text(encoding="utf-8"))
        merge_pins(pins, {str(receipt): digest})
        merge_pins(pins, data["validated_hashes"])
        merge_pins(pins, {str(folder / p): h for p, h in data["artifact_hashes"].items()})
        receipts[name], folders[name] = data, folder
    for path in [Path(__file__), Path("tests/test_av_slot_weight_publication.py")]:
        merge_pins(pins, {str(path.resolve()): sha(path)})
    verify(pins)
    primary, replay, smoke = [receipts[k] for k in ["primary", "replay", "smoke"]]
    admit(primary, replay, smoke)
    if primary["validated_hashes"] != replay["validated_hashes"]:
        raise ValueError("replay physical profile differs")
    for key in REPLAY_KEYS:
        if primary[key] != replay[key]:
            raise ValueError("replay numeric report differs: " + key)
    for name in primary["artifact_hashes"]:
        a, b = folders["primary"] / name, folders["replay"] / name
        if name.endswith(".npz"):
            with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
                if set(left.files) != set(right.files) or any(
                    not np.array_equal(left[k], right[k]) for k in left.files
                ):
                    raise ValueError("replay prediction arrays differ")
        elif sha(a) != sha(b):
            raise ValueError("replay weight/history/config bytes differ")
    history = [
        json.loads(line)
        for line in (folders["primary"] / "training-history.jsonl").read_text().splitlines()
    ]
    calls = 1 + sum(h["backtracks"] + 1 if h["accepted"] else 24 for h in history[1:])
    rows = primary["training_accounting"]["actual_train_rows_used"]
    if (
        len(history) != 513
        or [h["iteration"] for h in history] != list(range(513))
        or primary["objective_gradient_evaluations"] != calls
        or primary["objective_label_presentations"] != calls * rows
        or primary["iteration_role_presentations"] != 512 * rows
        or history[-1]["weights"] != primary["weights"]
    ):
        raise ValueError("history/accounting mismatch")
    for policy, report in primary["policies"].items():
        for split in ["train", "calibration", "validation"]:
            with np.load(folders["primary"] / f"{policy}-{split}.npz", allow_pickle=False) as saved:
                measured = choice_metrics(
                    saved["probabilities"], saved["targets"], saved["log_probabilities"]
                )
                if measured != report["calibrated"][split]:
                    raise ValueError("saved prediction metrics differ")
    outcome = decision(primary)
    summary = dict(
        status="PASS_HASH_BOUND_SLOT_WEIGHT_PUBLICATION",
        source_commit=primary["source_commit"],
        receipt_hashes={
            str(getattr(args, n) / "result.json"): getattr(args, n + "_sha256") for n in receipts
        },
        outcome=outcome,
        runs={
            name: {k: v for k, v in data.items() if k != "validated_hashes"}
            for name, data in receipts.items()
        },
        actual_fresh_fit_and_all_payloads_exact=True,
    )
    lines = [
        "# Fixed TRAIN-only AV slot reliability weights",
        "",
        "Execution and exact fresh-process refit PASS; quality adoption gates are separate.",
        "Only six nonnegative coefficients (sum6) are fitted; encoders, original heads,",
        "media, catalogs, native51 options and25 letters remain unchanged. No GPU,",
        "base-model updates, cloud cost or sealed-audit access.",
        "",
        f"Source commit: `{primary['source_commit']}`.",
        "",
        "## Fixed fit and accounting",
        "",
        "512 projected-gradient full-batch iterations; fixed final point, no development",
        "checkpoint selection. TRAIN-only weights saved before calibration; the same",
        "17-temperature grid chooses query temperature before development scoring.",
        f"Query temperature: {primary['query_temperature']:.12g}.",
        "",
        f"Weights in {primary['slot_order']} order: `{primary['weights']}`.",
        "",
        f"{rows} roles/batch over4080 physical parents;32248 unique query/candidate",
        "contexts and392 repeated contexts/batch across both policies. Source/speaker/",
        "media/normalized-content gates and original sample order are preserved.",
        f"Iteration-level presentations: {primary['iteration_role_presentations']};",
        f"actual objective/gradient evaluations: {calls}; objective label uses:",
        f"{primary['objective_label_presentations']}. These are repeated uses, not new assets.",
        "398 calibration /889 development parents;36/4/9 speaker groups.",
        "",
        "AV slice122790905 registered parameters/coefficients includes both full heads",
        "and six added scalars (five free simplex degrees). This is not a complete",
        "text/image/audio/video resident product or its package-size result.",
        "",
        "## TRAIN-only raw learning curve",
        "",
        "| Iteration | Random Acc | Random NLL | Hard Acc | Hard NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for iteration, values in primary["train_curves"].items():
        lines.append(
            f"| {iteration} | {values['random']['accuracy']:.8f} | "
            f"{values['random']['nll']:.8f} | {values['hard']['accuracy']:.8f} | "
            f"{values['hard']['nll']:.8f} |"
        )
    for policy, report in primary["policies"].items():
        lines += [""]
        table_metrics(lines, policy, report)
        lines += [
            f"Calibrated train-minus-development Accuracy gap: "
            f"{outcome['policies'][policy]['train_minus_validation_accuracy_pp']:.8f}pp.",
            f"Prospective policy eligibility: {report['eligible_for_runtime_trial']}.",
            "",
        ]
    lines += [
        "## Decision and limitations",
        "",
        f"`{outcome['action']}`.",
        "",
        "Both policies must meet the frozen gate. Random Accuracy and Brier worsen,",
        "and its NLL improvement is below1%; the comparison is closed, parent retained.",
        "Hard Accuracy gains only one correlated role; its paired Accuracy and NLL",
        "intervals include zero. Distance1 hard stratum has86 roles and loses4/86",
        "correct answers; it is reported despite falling below the100-role gate.",
        "Training NLL descent does not establish development improvement or a model",
        "capacity ceiling. No extra steps/seeds, per-policy fits, coefficient search",
        "or runtime/product promotion are justified by this result.",
        "",
        "Existing native runtime stays on equal weights. Four-modality95–98%,",
        "independent task/OOD quality and full resident integration remain unmet.",
        "",
        "## Actual execution and artifacts",
        "",
    ]
    for name, data in receipts.items():
        lines += [
            f"- {name}: `{getattr(args, name)}`, `{data['status']}`;",
            f"  {data['seconds']:.6f}s, peak CPU {data['peak_cpu_bytes']} bytes;",
            f"  `{data['started_utc']}` to `{data['ended_utc']}`;",
            "  command: `python -m scripts.train_lombard_correspondence_weights "
            + " ".join(data["command"][1:])
            + "`.",
        ]
    lines += [
        "",
        "Full primary/replay weights, every iteration, raw/calibrated metrics,",
        "per-example IDs/targets/probabilities/log probabilities, controls and strata",
        "are retained outside Git. Small JSON includes all attempted run summaries,",
        "payload hashes, effective protocol/source revisions and uncertainty results.",
        "Terminal publication receipt checks input and source hashes before/after.",
        "CPU/NoTorch/Ruff/manifests and exact pushed-commit CI are recorded separately.",
        "",
    ]
    docs = Path("docs/pretrained_reuse")
    paths = [docs / f"R2_AV_SLOT_WEIGHTS_RESULTS_20261015{suffix}" for suffix in [".md", ".json"]]
    if any(p.exists() for p in paths):
        raise FileExistsError("new publication outputs required")
    verify(pins)
    paths[0].write_text("\n".join(lines), encoding="utf-8")
    paths[1].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    verify(pins)
    final = dict(
        status=summary["status"],
        validated_hashes=pins,
        output_hashes={str(p.resolve()): sha(p) for p in paths},
        eligible_for_runtime_trial=outcome["eligible_for_runtime_trial"],
    )
    (args.output / "result.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(final["status"])


if __name__ == "__main__":
    main()
