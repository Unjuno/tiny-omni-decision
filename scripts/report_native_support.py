"""Build compact measured tables after the complete fixed support run and replay."""

import json
from pathlib import Path

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-support-comparison-01"
r = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
assert r["status"] == "PASS_SUPPORT_EXECUTION_AND_REPLAY_AUDIT"
run = r["run"]
selected = r["comparisons"]["selected"]
lines = [
    "# Native Decision: positive-support comparison (2026-10-16)",
    "",
    "## Question and execution",
    "",
    "The pipeline audit found that available TRAIN positives were absent from the",
    "actual consumed STARSS subset. This experiment tests one predefined repair",
    "at the same long budget; it does not assume more positive examples must help.",
    "One fresh seed17 run completed32,768 updates and all scheduled validations.",
    "Best/final predictions and the original control predictions on its TRAIN",
    "subset reproduced bit-exactly in a separate process. No product promotion.",
    "[Pre-registered protocol](R2_NATIVE_SUPPORT_PROTOCOL_20261016.md).",
    "",
    "## Fixed comparison",
    "",
    "Same frozen encoders/features, native readout889,345 parameters, total",
    "196,254,874 parameters before quantization. Same initial weights, AdamW,",
    "LR0.001, warmup62, cosine32,768, weight decay0.01, gradient clip1, eight",
    "microbatches/update, option-shuffle rule and full24,346-question validation.",
    "No early stop. Selector remains task/source/modality macro NLL, min_delta1e-5,",
    "earliest ties and initial0 eligible. No architecture/frame/crop change.",
    "",
    "Only292 unique negative STARSS rows are replaced with unused positive TRAIN",
    "rows of the same question/candidate condition. Each replacement retains its",
    "32 occurrence slots:9,344 presentations change; all229,376 non-STARSS slots",
    "are identical. Total6,656 unique questions and262,144 presentations stay fixed.",
    "STARSS positive unique support rises110→402 out of1,024 questions. Preserve",
    "all original positives, including speech70/85. Candidate shuffle still hashes",
    "row identity, so replaced rows use their own original-rule permutations.",
    "",
    "This intervention combines positive support, recording diversity and label",
    "prior changes. It cannot isolate these effects from one another. The floor",
    "is an engineering choice, not an optimized ratio. No validation labels chose",
    "replacement questions. No rare class or difficult validation example was removed.",
    "",
    "| Event | Unique quota | Positive before | Positive after | Available TRAIN positives | "
    "Positive recordings after | Positive rooms after |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, v in r["support_events"].items():
    lines.append(
        f"| {name} | {v['unique_quota']} | {v['before_positive_unique']} | "
        f"{v['after_positive_unique']} | {v['available_positive_unique']} | "
        f"{v['after_support']['positive_recordings']} | {v['after_support']['positive_rooms']} |"
    )
lines += [
    "",
    "Bell remains unsupported in admitted TRAIN. Water tap has no positive",
    "validation examples. Neither limitation is hidden by aggregate Accuracy.",
    "",
    "## Full learning curve",
    "",
    "TRAIN probe covers all6,656 candidate-consumed questions; its membership",
    "differs from control for STARSS, so the two TRAIN means are not a",
    "matched-question comparison. Separate6,948-question union results are saved.",
    "",
    "| Step | TRAIN macro Acc | Val macro Acc | Val macro NLL | Text | Image | Audio | Video | "
    "Non-speech balanced Acc | Best step |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for h in r["curve"]:
    v = h["validation"]
    ev = [
        x["balanced_accuracy"]
        for k, x in v["event_classes"].items()
        if k != "speech" and x["balanced_accuracy"] is not None
    ]
    fields = [
        str(h["step"]),
        f"{h['train_probe']['macro']['accuracy']:.2%}",
        f"{v['macro']['accuracy']:.2%}",
        f"{v['macro']['nll']:.5f}",
    ]
    fields += [
        f"{v['modality_macro'][m]['accuracy']:.2%}" for m in ("text", "image", "audio", "video")
    ]
    fields += [f"{sum(ev) / len(ev):.2%}", str(h["best_step"])]
    lines.append("| " + " | ".join(fields) + " |")
lines += [
    "",
    "![Complete fixed-budget curves](R2_NATIVE_SUPPORT_RESULTS_20261016.png)",
    "",
    "## Selected and equal-budget comparisons",
    "",
    "The selected comparison uses each run's unchanged macro-NLL rule, not",
    "the checkpoint with the best STARSS or aggregate Accuracy. Equal-budget",
    "8,192 and32,768 points are predeclared secondary comparisons.",
    "",
    "| Contrast | Control step | Candidate step | Macro NLL control → candidate | "
    "Macro Acc control → candidate | Non-speech balanced Acc control → candidate |",
    "|---|---:|---:|---|---|---|",
]
for label, c in r["comparisons"].items():
    a, b = c["control"], c["candidate"]
    lines.append(
        f"| {label} | {c['control_step']} | {c['candidate_step']} | "
        f"{a['macro']['nll']:.5f} → {b['macro']['nll']:.5f} | "
        f"{a['macro']['accuracy']:.2%} → {b['macro']['accuracy']:.2%} | "
        f"{c['non_speech_control']['mean']:.2%} → {c['non_speech_candidate']['mean']:.2%} |"
    )
lines += [
    "",
    "| Checkpoint | Modality | Accuracy | NLL | Brier | ECE |",
    "|---|---|---:|---:|---:|---:|",
]
for label, m in (
    (f"Control selected{selected['control_step']}", selected["control"]),
    (f"Candidate selected{selected['candidate_step']}", selected["candidate"]),
    ("Control final32768", r["comparisons"]["equal32768"]["control"]),
    ("Candidate final32768", r["comparisons"]["equal32768"]["candidate"]),
):
    for mod in ("text", "image", "audio", "video"):
        v = m["modality_macro"][mod]
        lines.append(
            f"| {label} | {mod} | {v['accuracy']:.2%} | {v['nll']:.5f} | "
            f"{v['brier']:.5f} | {v['ece']:.5f} |"
        )
lines += [
    "",
    "Hierarchical macro metrics balance tasks within source and sources within",
    "modality; ordinary sample-weighted metrics and every task are also saved.",
    "No modality or required-task95–98% gate is relaxed.",
    "",
    "## Event detection and retention",
    "",
    "![All event classes](R2_NATIVE_SUPPORT_EVENTS_20261016.png)",
    "",
    "Non-speech macro balanced Accuracy includes all ten classes with both",
    "validation labels, including unsupported bell. Water-tap recall/balanced",
    "Accuracy is undefined, rather than assigned zero or silently removed.",
    "",
    "| Event | Val positives | Val negatives | Recall control → candidate | "
    "Specificity control → candidate | Balanced Acc control → candidate |",
    "|---|---:|---:|---|---|---|",
]


def percent(value):
    return "undefined" if value is None else f"{value:.2%}"


for event, a in selected["control"]["event_classes"].items():
    b = selected["candidate"]["event_classes"][event]
    fields = [event, str(a["positive_count"]), str(a["negative_count"])]
    fields += [
        percent(a[k]) + " → " + percent(b[k])
        for k in ("recall", "specificity", "balanced_accuracy")
    ]
    lines.append("| " + " | ".join(fields) + " |")
lines += ["", "| Research retention check | Result |", "|---|---|"]
for k, v in r["research_retention"].items():
    if isinstance(v, bool):
        lines.append(f"| {k} | {v} |")
lines += [
    "",
    "The2pp decline flag is for research retention; smaller declines are still",
    "trade-offs and do not satisfy product quality gates. Selected accuracy",
    "differences (candidate minus control): "
    + ", ".join(
        f"{k} {v * 100:+.3f}pp" for k, v in r["research_retention"]["accuracy_deltas"].items()
    )
    + ".",
    "",
    "## Paired uncertainty",
    "",
    "STARSS overall metrics:5,000 paired recording-cluster bootstrap resamples,",
    "seed17, linear percentile95% intervals. Every question/window from a",
    "sampled recording stays together. These are development diagnostics with",
    "model-selection bias, not independent audit confidence intervals.",
    "Recording clusters do not account for dependence between recordings in the same room.",
    "",
    "| Contrast | Metric | Candidate minus control | 95% interval |",
    "|---|---|---:|---|",
]
for label, c in r["comparisons"].items():
    for metric in ("accuracy", "nll", "brier"):
        x = c["starss_paired_bootstrap"][metric]
        lines.append(
            f"| {label} | {metric} | {x['candidate_minus_control']:+.6f} | "
            f"[{x['low']:+.6f}, {x['high']:+.6f}] |"
        )
lines += [
    "",
    "## Same-question TRAIN union",
    "",
    "The fixed union contains the original6,656 questions plus292 new positives.",
    "Control best/final and candidate final are evaluated on exactly these6,948",
    "questions. Parent original-subset logits match saved parent outputs exactly.",
    "Original versus candidate consumed sets differ; union results show fit",
    "capacity on a common question list and are not an independent test.",
    "",
    "| Model | Union macro Acc | Union macro NLL | STARSS Acc | STARSS NLL |",
    "|---|---:|---:|---:|---:|",
]
for name, metrics in r["train_union"].items():
    m = metrics["union"]
    lines.append(
        f"| {name} | {m['macro']['accuracy']:.2%} | {m['macro']['nll']:.5f} | "
        f"{m['sources']['starss']['accuracy']:.2%} | {m['sources']['starss']['nll']:.5f} |"
    )
lines += [
    "",
    "## Supplementary condition diagnostic",
    "",
    "Post-hoc analysis of saved predictions, excluded from selection and retention.",
    "For each audio window, preserve all12 event questions. Measure exact agreement",
    "on the full12-answer vector and whether windows with both positive/negative",
    "targets receive identical binary predictions across all questions. Within-window",
    "event ranking compares yes probabilities across present versus absent classes.",
    "Uniform answers alone do not prove query inputs were ignored, and cross-question",
    "scores have not been calibrated; these are descriptive diagnostics.",
    "",
    "| Model | Windows | Mixed-target windows | Full-vector Accuracy | "
    "Uniform predictions on mixed targets | Within-window event ranking AUC |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, value in selected["posthoc_condition_diagnostic"].items():
    lines.append(
        f"| {name} | {value['observations']} | {value['mixed_target_observations']} | "
        f"{value['all_event_questions_accuracy']:.2%} | "
        f"{value['mixed_target_uniform_prediction_fraction']:.2%} | "
        f"{value['mixed_target_mean_event_ranking_auc']:.5f} |"
    )
lines += [
    "",
    "## Interpretation",
    "",
    "INTERPRETATION_PENDING_COMPLETE_RESULTS",
    "",
    "## Exposure, environment and artifacts",
    "",
    "| Scope | Presentations | Unique questions | Repeats | Media assets | Groups |",
    "|---|---:|---:|---:|---:|---:|",
]
for label, key in (
    ("Full run", "actual_accounting"),
    (f"Selected{run['best_step']}", "selected_accounting"),
):
    a = run[key]
    lines.append(
        f"| {label} | {a['consumed']} | {a['unique_questions']} | {a['repeated_questions']} | "
        f"{a['unique_media_assets']} | {a['unique_groups']} |"
    )
lines += [
    "",
    "| Phase | Wall seconds | Train seconds | Eval seconds | Peak allocated CUDA bytes |",
    "|---|---:|---:|---:|---:|",
]
for phase, v in r["resources"].items():
    lines.append(
        f"| {phase} | {v['seconds']:.2f} | {v['training_seconds']:.2f} | "
        f"{v['evaluation_seconds']:.2f} | {v['peak_vram_bytes']} |"
    )
lines += [
    "",
    f"Total local process-body wall:{sum(x['seconds'] for x in r['resources'].values()):.2f}s. "
    "RTX3080 Laptop; cloud cost0; FP32 and TF32 off. Frozen-feature measurements",
    "exclude the historical encoder extraction cost. Exact dependency versions,",
    "arguments, timestamps, source/input hashes and per-update LR/gradients are saved.",
    "Eval seconds cover inference loops; metric aggregation/checkpoint I/O are in total wall.",
    "CUDA memory here measures cached-feature readout training, not the full encoder runtime.",
    "The last scheduled optimizer call has zero LR; it consumes examples without a weight update.",
    "",
    f"Training source:`{run['source_commit']}`.",
    f"Effective config SHA256:`{run['config_sha256']}`.",
    "Config:`configs/pretrained_reuse/native_support_v1.json`.",
    "Runner:`scripts/train_native_support.py`; analyze/plot/report sources",
    "are published from their exact executed bytes after the run/replay ends.",
    "",
    "Artifacts:`C:/CodexArtifacts/pretrained-reuse-r2/native-support-{plan,smoke,fit,replay,comparison,verification}-01/`.",
    "The best and final weights/checkpoints remain separate. Optimizer/RNG",
    "states are saved, but no resume API or exact-resume guarantee is claimed.",
    f"All{r['preserved_artifact_count']} pinned old artifacts are unchanged.",
    "The original Teacher/QAT/corpora are untouched; calibration and sealed",
    "audit payloads were not opened. No new cloud compute or model downloads.",
    "Preflight Ruff,2,135 CPU tests (1 skipped) and25 manifest commands passed.",
    "Post-result verification and exact publication-commit CI are recorded in",
    "the verification directory. The overall project goal remains incomplete.",
    "",
]
lines += [
    "## Executed commands",
    "",
    "Working directory:`C:/CodexWorktrees/pretrained-reuse-omni`.",
    "PowerShell environment (same installed dependencies, no upgrades):",
    "",
    "```powershell",
    "$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;"
    "C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;"
    "C:/CodexWorktrees/pretrained-reuse-omni/src;C:/CodexWorktrees/pretrained-reuse-omni'",
    "$env:OMP_NUM_THREADS='4'",
    "Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue",
    "```",
    "",
]
for phase, directory, reference in (
    ("smoke", "smoke", None),
    ("train", "fit", "smoke"),
    ("replay", "replay", "fit"),
):
    command = (
        f"python -u -m scripts.train_native_support --phase {phase} --policy support "
        f"--output C:/CodexArtifacts/pretrained-reuse-r2/native-support-{directory}-01"
    )
    if reference:
        command += (
            " --reference C:/CodexArtifacts/pretrained-reuse-r2/"
            f"native-support-{reference}-01 --reference-sha256 "
            + r["resources"][reference]["receipt_sha256"]
        )
    lines += ["```powershell", command, "```", ""]
lines += [
    "Existing output directories are immutable. These exact historical commands are",
    "recorded for reproduction; they intentionally refuse to overwrite an existing run.",
    "",
]
(OUT / "report-draft.md").write_text("\n".join(lines), encoding="utf-8")


def compact_metrics(m):
    return {
        k: m[k]
        for k in (
            "overall",
            "macro",
            "modality_macro",
            "source_macro",
            "minimum_modality_accuracy",
            "event_classes",
        )
    }


public = {
    k: v
    for k, v in r.items()
    if k not in ("curve", "parent_curve", "train_union", "endpoint_state_membership")
}
for key in ("curve", "parent_curve"):
    public[key] = [
        dict(
            step=h["step"],
            train_probe=compact_metrics(h["train_probe"]),
            validation=compact_metrics(h["validation"]),
            selected=h["selected"],
            best_step=h["best_step"],
            warnings=h["warnings"],
        )
        for h in r[key]
    ]
public["train_union"] = {
    name: {subset: compact_metrics(m) for subset, m in groups.items()}
    for name, groups in r["train_union"].items()
}
public["full_comparison_artifact"] = str(OUT / "comparison.json")
(OUT / "public-summary.json").write_text(
    json.dumps(public, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print("Draft tables saved; interpretation must use completed results.")
