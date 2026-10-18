"""Render only verified temporal evidence; no model execution or metric selection."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
DOCS = Path("docs/pretrained_reuse")
PREFIX = "R2_LOMBARD_TEMPORAL_RESULTS_20261015"
EXPECTED = {
    "lombard-temporal-smoke-01": "2676b88d2b0c44986986b457688ce24cf0b4812b32b9cc297df3331caa1b2f52",
    "lombard-temporal-full-01": "09ef033c461313a84d34a7db5b6a754c57619871ba1ea9e7f9bdb746e830ec0e",
    "lombard-temporal-evaluation-01": (
        "c97d975df27f4931be0b256e5a0ed90f476b2ba7dbbc7534085b7667f3694fdc"
    ),
}


def read(name):
    path = ROOT / name / "result.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED[name]:
        raise ValueError("fixed executed receipt hash differs")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    smoke, fit, audit = (read(n) for n in EXPECTED)
    if (
        smoke["status"] != "PASS_FIVE_ARM_TRAIN_ONLY_SMOKE"
        or fit["status"] != "PASS_FIVE_ARM_TRAINING"
        or audit["status"] != "PASS_FRESH_REPLAY_CONTROLS_AND_GROUPED_COMPARISON"
    ):
        raise ValueError("all executed stages must pass")
    changed = {}
    for arm in ["joint_broadcast", "joint_ordered"]:
        base = ROOT / "lombard-temporal-evaluation-01"
        with np.load(base / f"{arm}-matched.npz", allow_pickle=False) as saved:
            original = saved["probabilities"].copy()

        def pred(value):
            return np.stack([z.argmax(1) for z in np.split(value, [4, 8, 12, 37, 47], 1)], 1)

        changed[arm] = {}
        for condition in ["video_mismatch", "reverse"]:
            with np.load(base / f"{arm}-{condition}.npz", allow_pickle=False) as saved:
                changed[arm][condition] = int(
                    (pred(saved["probabilities"]) != pred(original)).sum()
                )
    report = dict(
        status="EXECUTED_AND_REPLAYED_NOT_OMNI_PRODUCT_COMPLETION",
        receipt_hashes=EXPECTED,
        paths={n: str(ROOT / n) for n in EXPECTED},
        config=fit["config"],
        effective_config_sha256=fit["effective_config_sha256"],
        config_sha256=fit["validated_hashes"][
            str(Path("configs/pretrained_reuse/lombard-grid-temporal-v1.json").resolve())
        ],
        source_commit=fit["source_commit"],
        provenance=fit["provenance"],
        started_utc=fit["started_utc"],
        ended_utc=fit["ended_utc"],
        commands={"smoke": smoke["command"], "fit": fit["command"], "evaluate": audit["command"]},
        stages={
            n: {k: r[k] for k in ["status", "seconds", "peak_cpu_bytes", "output_bytes"]}
            for n, r in zip(EXPECTED, [smoke, fit, audit], strict=True)
        },
        environment=dict(
            cpu="Intel Core i7-12700H,14cores/20threads; process4threads/interOp1",
            device="cpu",
            dtype="float32",
            deterministic=True,
            python=fit["python"],
            torch=fit["torch"],
            numpy=smoke["numpy"],
            gpu_used=False,
            cloud_cost=0,
            note="CPU tests overlapped part of training; wall time is observed shared-PC time",
        ),
        arms=fit["arms"],
        controls=audit["controls"],
        intervals=audit["intervals"],
        control_slot_answer_changes=changed,
        replay_checks=audit["arms"],
        smoke_timing_probe=smoke["timing_probe"],
        scope=(
            "Observed nine-speaker development, native six-slot command task,"
            "51options; no blind audit"
        ),
        interpretation={
            "video_only": "Ordered frame information improves Accuracy/NLL/Brier; ECE worsens",
            "joint_temporal": "No useful ordered-over-broadcast gain established",
            "joint_gain": (
                "Scores improve versus audio, but mismatched video changes no answers; "
                "matched visual semantics unproved"
            ),
            "adoption": "No product promotion or general95-98% Omni claim",
            "next": (
                "Diagnose visual pathway versus constant/mask contribution with frozen "
                "models before further fusion fitting"
            ),
        },
    )
    result_path = DOCS / (PREFIX + ".json")
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    curve = DOCS / "R2_LOMBARD_TEMPORAL_CURVES_20261015.csv"
    with curve.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "arm",
                "epoch",
                "selected",
                "consumed_assets",
                "split",
                "slot",
                "count",
                "accuracy",
                "nll",
                "brier",
                "ece",
                "mean_confidence",
            ]
        )
        for arm, run in fit["arms"].items():
            for row in run["history"]:
                for split in ["train_fixed", "validation"]:
                    for slot, metric in {
                        "macro": row[split]["macro"],
                        **row[split]["by_slot"],
                    }.items():
                        writer.writerow(
                            [
                                arm,
                                row["epoch"],
                                row["epoch"] == run["best_epoch"],
                                row["consumed_assets"],
                                split,
                                slot,
                                *[
                                    metric[k]
                                    for k in [
                                        "count",
                                        "accuracy",
                                        "nll",
                                        "brier",
                                        "ece",
                                        "mean_confidence",
                                    ]
                                ],
                            ]
                        )
    lines = [
        "# Native temporal readout: completed comparison and limits",
        "",
        "All five fresh CPU fits completed24epochs/768updates and passed separate-process",
        "best/final train+validation replay, calibration, saved metrics/sampling/optimizer",
        "and frozen-source checks. This is a six-slot speech-command task, not general",
        "Omni quality. All51options, including25letters, and4080/398/889 grouped utterances",
        "are retained. Nine development speakers and checkpoint selection limit inference.",
        "",
        "## Result and decision",
        "",
        "Retaining chronological visual information improves video-only macro Accuracy",
        "29.85%→48.01%, NLL1.734875→1.377506 and Brier .764713→.609485.",
        "Paired whole-speaker95% Accuracy difference is+18.17pp[13.79,21.47].",
        "ECE worsens .027221→.043730; the weakest slot(letter) remains6.97%.",
        "This changes temporal variation as well as order; not a pure order-only effect.",
        "",
        "Joint broadcast/ordered reach99.53%/99.51% macro Accuracy, minimum slot98.20%",
        "and complete-command97.19%/97.08%. **Mismatching video changes0/5334 slot",
        "answers in either arm**, with nearly unchanged NLL. The higher score versus",
        "audio98.18% does not establish matched audiovisual semantic benefit. A learned",
        "constant visual state, extra effective parameterization or optimization could",
        "contribute; those explanations are not yet measured. Audio-only's absent visual",
        "state is forced zero, so equal registered parameters do not imply equal active",
        "capacity. Do not label this a successful joint Omni model or promote it.",
        "",
        "Ordered-minus-broadcast joint Accuracy−.01875pp[−.05624,0], NLL interval",
        "includes no change. More temporal readout capacity/budget is not justified by",
        "this comparison alone. Keep all results; next diagnose fixed visual-state/mask",
        "contributions before another fusion fit. Video-only quality remains inadequate.",
        "",
        "## Selected calibrated checkpoints",
        "",
        "| Arm | Best epoch | Accuracy | NLL | Brier | ECE | Min slot | All-six correct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, run in fit["arms"].items():
        m, b = run["best_calibrated"], run["best_epoch"]
        k = m["macro"]
        lines.append(
            f"| {arm} | {b} | {k['accuracy']:.2%} | {k['nll']:.6f} | {k['brier']:.6f} | "
            f"{k['ece']:.6f} | {m['minimum_slot_accuracy']:.2%} | "
            f"{m['whole_utterance_accuracy']:.2%} |"
        )
    lines += [
        "",
        "## Per-slot calibrated metrics",
        "",
        "| Arm | Slot | Count | Accuracy | NLL | Brier | ECE | Mean confidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, run in fit["arms"].items():
        for slot, m in run["best_calibrated"]["by_slot"].items():
            lines.append(
                f"| {arm} | {slot} | {m['count']} | {m['accuracy']:.2%} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece']:.6f} | {m['mean_confidence']:.6f} |"
            )
    lines += [
        "",
        "## Fixed selected controls",
        "",
        "| Arm | Control | Accuracy | NLL | Brier | ECE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, values in audit["controls"].items():
        for name, metric in values.items():
            m = metric["macro"]
            lines.append(
                f"| {arm} | {name} | {m['accuracy']:.2%} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece']:.6f} |"
            )
    lines += [
        "",
        "Reversal means16cached visual tokens, not raw-frame re-encoding. Broadcast",
        "is reversal-invariant by construction. Ordered reversal changes34slot answers",
        "and worsens Accuracy/NLL; this alone does not prove matching video semantics.",
        "Donor controls retain p/l condition and exact donor marginal with different speakers.",
        "Absent modality uses unseen masks(OOD); poor scores do not quantify natural missingness.",
        "Audio mismatch is highly confident and wrong: calibration does not transfer to this",
        "artificial contradictory-input control. No controls were used for model selection.",
        "",
        "## Learning curves, selection and consumption",
        "",
        "All120epoch observations, train/validation per-slot Accuracy/NLL/Brier/ECE are in",
        "[curve CSV](R2_LOMBARD_TEMPORAL_CURVES_20261015.csv), with raw/final and grouped",
        "intervals in [complete small JSON](R2_LOMBARD_TEMPORAL_RESULTS_20261015.json).",
        "Earliest minimum unrounded raw validation macro NLL chooses best; final24is",
        "saved separately. Calibration uses only398independent rows and the fixed17-point",
        "temperature grid. Same106355head, seed17initialization/sample orders, batch128,",
        "fixed AdamW LR/WD .001, no encoder training/noise/source/frame changes.",
        "",
        "| Arm | Selected consumed assets | Total assets | Unique assets | "
        "Raw best NLL | Raw final NLL | Final train Acc | Final val Acc |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, run in fit["arms"].items():
        lines.append(
            f"| {arm} | {run['selected_consumed_assets']} | 97920 | 4080 | "
            f"{run['best_raw']['macro']['nll']:.6f} | {run['final_raw']['macro']['nll']:.6f} | "
            f"{run['history'][-1]['train_fixed']['macro']['accuracy']:.2%} | "
            f"{run['final_raw']['macro']['accuracy']:.2%} |"
        )
    lines += [
        "",
        "Each arm consumes587520slot decisions,24480unique decisions and93840repeated",
        "asset presentations. Five arms together489600presentations share4080unique",
        "assets; do not count five runs as unique data. The smoke adds40training presentations",
        "over4unique assets/10updates. Its gradient-only timing probe separately adds384",
        "presentations/0updates and changes no weights. It is not part of main-run consumption.",
        "Video broadcast strongly overfits after best5; ordered improves through18then",
        "train NLL continues falling while development NLL worsens, even as Accuracy rises.",
        "Joint23→24has slight development NLL deterioration; audio still improves through24.",
        "Do not automatically choose last or extend budgets. Frozen-source pretraining",
        "overlap remains UNKNOWN. Old pooled results are preserved and are not an equal-capacity",
        "causal comparison. Bootstrap10000/seed17whole-speaker/linear percentile95%,",
        "nine groups, single seed and observed-development selection—not blind inference.",
        "",
        "## Resources, reproducibility and preservation",
        "",
        f"Source commit `{fit['source_commit']}`. All frozen physical files/runtime pins",
        "and complete training artifact hashes pass before/terminal rechecks. The smaller",
        "result JSON records exact commands, source/split/cache/config/report hashes and times.",
        "Artifacts remain outside Git under`C:/CodexArtifacts/pretrained-reuse-r2/`:",
    ]
    for name, result in zip(EXPECTED, [smoke, fit, audit], strict=True):
        lines.append(
            f"- `{name}`: {result['seconds']:.3f}s, "
            f"peak {result['peak_cpu_bytes'] / 2**30:.3f}GiB, "
            f"receipt SHA`{EXPECTED[name]}`."
        )
    lines += [
        "",
        "Intel i7-12700H, CPU4threads/interOp1, FP32/deterministic, Python3.11.9,",
        "Torch2.6.0+cu124, NumPy2.4.6. GPU unused/encoders never loaded in fitting; no",
        "cloud cost. CPU tests overlapped part of fitting; wall time is shared-PC time.",
        "7200s/3GiB/128MiB/1GiB reserve limits held. Existing GPU8GiB guard/interrupted",
        "run, v1/v2Teacher weights/config/corpora, source caches and sealed data untouched.",
        "Best/final weights, optimizer/global step/RNG/orders retained; no verified strict resume.",
        "Separate evaluator keeps original requires_grad flags for numeric parity but runs",
        "eval/no_grad without optimizer or updates and checks unchanged model tensors.",
        "",
        "Native encoders122573568 plus106355head/2304normalization =122682227resident",
        "parameters for this AV composition. Deployment encoder costs remain included;",
        "not a packaged four-modality model/size/latency proof. JSON native-option reorder",
        "checks pass all889selected observations per arm. Weights/media are not added to Git.",
        "",
        "**Remaining:** genuine input-dependent AV benefit, natural-noise/task/OOD transfer,",
        "all four modalities, fresh audit and complete <=200M deployment. The goal remains active.",
    ]
    (DOCS / (PREFIX + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Rendered verified five-arm results and120point curves; no training or promotion")


if __name__ == "__main__":
    main()
