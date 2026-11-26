"""Generate a factual full-extent comparison report from the verified result."""

import json
from pathlib import Path

OUT = Path("C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-comparison-01")
r = json.loads((OUT / "comparison.json").read_text())
assert r["status"] == "PASS_FULL_EXTENT_EXECUTION_AND_REPLAY_AUDIT"
run = r["run"]
selected = r["comparisons"]["selected"]
lines = [
    "# Native full-extent preprocessing: measured comparison",
    "",
    "**Execution: PASS. Product goal: incomplete.** One fixed seed17 run completed32,768 "
    "updates; fresh-process best/final prediction replay is bit-exact. This is observed "
    "development validation, not a blind audit.",
    "",
    "## Intervention and integrity",
    "",
    "Replace only Image/Video resize-short-side/center-crop with bilinear full-extent "
    "fit into288x288 and centered black padding. Image640x480→288x216; "
    "Video480x320→288x192. Object scale is reduced and padding changes; these effects "
    "cannot be separated from restored spatial coverage by this comparison.",
    "",
    "This registered comparison changes one shared visual preprocessing policy in both Image "
    "and Video. Because the readout is trained jointly, per-modality differences cannot isolate "
    "an Image-only versus Video-only causal contribution. Earlier audit suggestions to separate "
    "those routes are not claimed to have been completed by this combined policy experiment.",
    "",
    "Keep original corpus/questions/targets/options,8 video frames, frozen encoder weights, "
    "queries, Text/Audio features,889,345-parameter readout, initialization, sampler, "
    "option permutations and32,768-update LR schedule. Total prequantization inference "
    "parameters196,254,874. No failed support policy, extra seeds, encoder tuning, "
    "calibration, sealed audit, quantization or product promotion.",
    "",
    "Original-feature CPU smoke parity was bit-exact. All8,936 images and456 videos "
    "were extracted into new458 visual banks. Complete media hashes, observation indices, "
    "source pins and smoke/full parity passed. Sampler, LR plan, TRAIN probe and validation "
    "ID/options/target files are byte-identical to control. Initial Text/Audio validation "
    "logits match control exactly. All subsequent probabilities are raw temperature1 softmax.",
    "",
    "Read-only upstream inspection confirms four temporal attention layers in the pinned "
    "VideoPrism base and interpolation of its16-frame temporal positional embeddings to8. "
    "The returned temporal/spatial tokens are retained; readout coordinates denote sampled "
    "order rather than physical synchronization. This comparison changes neither mechanism "
    "and does not test frame bandwidth or the incremental value of LSTM/xLSTM.",
    "",
    "The first encoder smoke failed before extraction because the version recorder queried "
    "`tensorflow` instead of installed `tensorflow_cpu`. Only metadata-name handling changed; "
    "the failed receipt is retained. No dependency or experiment condition changed.",
    "",
    "## Selected checkpoints",
    "",
    f"Control step{selected['control_step']}; full-extent step{selected['candidate_step']}. "
    "The unchanged hierarchical task/source/modality macro-NLL selector uses min_delta1e-5 "
    "and earliest tie, including step0. Final weights are kept separately.",
    "",
    "| Model / modality | Accuracy (%) | NLL | Brier | ECE |",
    "|---|---:|---:|---:|---:|",
]
for label, value in (("Control", selected["control"]), ("Full extent", selected["candidate"])):
    for mod, metrics in [("macro", value["macro"]), *value["modality_macro"].items()]:
        lines.append(
            f"| {label} / {mod} | {100 * metrics['accuracy']:.4f} | "
            f"{metrics['nll']:.6f} | {metrics['brier']:.6f} | {metrics['ece']:.6f} |"
        )
lines += [
    "",
    "## Predefined research retention rule",
    "",
    f"Retain for further research: **{r['retention']['retain_for_research']}**.",
]
for name, passed in r["retention"]["checks"].items():
    lines.append(f"- {name}: {passed}")
lines += [
    "",
    "This flag does not lower95–98% task gates. Every modality decline remains "
    "a measured trade-off, even if smaller than2pp.",
    "",
    "## Full learning curve",
    "",
    "| Step | Fixed TRAIN macro Acc (%) | Validation macro Acc (%) | Validation NLL | "
    "Image Acc (%) | Video Acc (%) | Temporal Video Acc (%) | Best step |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for h in r["curve"]:
    m = h["validation"]
    lines.append(
        f"| {h['step']} | {100 * h['train_probe']['macro']['accuracy']:.3f} | "
        f"{100 * m['macro']['accuracy']:.3f} | {m['macro']['nll']:.6f} | "
        f"{100 * m['modality_macro']['image']['accuracy']:.3f} | "
        f"{100 * m['modality_macro']['video']['accuracy']:.3f} | "
        f"{100 * m['tasks']['clevrer:temporal_descriptive']['accuracy']:.3f} | {h['best_step']} |"
    )
lines += [
    "",
    "![Full learning curves](R2_NATIVE_FULL_EXTENT_RESULTS_20261016.png)",
    "",
    "No performance early stopping. Rolling optimization metrics are not a fixed "
    "generalization gap. Full TRAIN/validation mixtures differ;43 shared-task gaps "
    "and every task/source/modality metric are retained in the JSON.",
    "",
    "### Equal-step differences (full extent minus control)",
    "",
    "| Comparison | Macro Acc (pp) | Macro NLL | Text Acc (pp) | Image Acc (pp) | "
    "Audio Acc (pp) | Video Acc (pp) |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for label, item in r["comparisons"].items():
    d = item["deltas"]
    mods = [100 * d["modalities"][m]["accuracy"] for m in ("text", "image", "audio", "video")]
    lines.append(
        f"| {label} | {100 * d['macro']['accuracy']:.3f} | {d['macro']['nll']:.6f} |  | ".join(
            f"{v:.3f}" for v in mods
        )
        + " |"
    )
lines += [
    "",
    "### Paired uncertainty at selected checkpoints",
    "",
    "| Subset | Assets | Questions | Accuracy delta (pp),95% interval | NLL delta,95% interval |",
    "|---|---:|---:|---|---|",
]
for label, item in selected["paired_visual_bootstrap"].items():
    a, n = item["accuracy"], item["nll"]
    lines.append(
        f"| {label} | {item['assets']} | {item['questions']} | "
        f"{100 * a['candidate_minus_control']:.3f} [{100 * a['low']:.3f}, {100 * a['high']:.3f}] | "
        f"{n['candidate_minus_control']:.6f} [{n['low']:.6f}, {n['high']:.6f}] |"
    )
lines += [
    "",
    "Paired underlying-image/scene bootstrap,5,000 replicates,seed17,linear "
    "percentile95% intervals. All questions from an asset stay together. Brier intervals "
    "and equal-step contrasts are in JSON. Development reuse, multiple comparisons and "
    "checkpoint selection limit inference; intervals are diagnostic.",
    "",
    "## Training consumption",
    "",
]
for label, item in (
    ("Whole run", run["actual_accounting"]),
    ("Selected checkpoint", run["selected_accounting"]),
):
    lines.append(
        f"- {label}: {item['consumed']:,} presentations; {item['unique_questions']:,} "
        f"unique questions; {item['repeated_questions']:,} repeats; "
        f"{item['unique_media_assets']:,} unique media assets; {item['unique_groups']:,} groups."
    )
lines += [
    "",
    "Unique questions by source: `"
    + json.dumps(r["unique_counts"]["source"], sort_keys=True)
    + "`.",
    "",
    "## Local resources and provenance",
    "",
    "| Stage | Wall seconds | Train seconds | Evaluation seconds | "
    "Peak allocated CUDA bytes | Peak process bytes |",
    "|---|---:|---:|---:|---:|---:|",
]
for label, item in r["resources"].items():
    lines.append(
        f"| {label} | {item['seconds']:.3f} | {item['training_seconds']:.3f} | "
        f"{item['evaluation_seconds']:.3f} | {item['peak_vram_bytes']} | "
        f"{item['peak_working_set_bytes']} |"
    )
for label, item in r["encoder"].items():
    lines.append(
        f"| Encoder {label} | {item['seconds']:.3f} | 0 | 0 | 0 | "
        f"{item.get('peak_working_set_bytes', 'UNKNOWN')} |"
    )
total = sum(v["seconds"] for v in r["resources"].values()) + sum(
    v["seconds"] for v in r["encoder"].values()
)
lines += [
    "",
    f"Total measured encoder/readout wall time {total:.3f}s ({total / 60:.2f}min); "
    "cloud cost0. CPU encoder extraction uses the original JAX0.6.2/TF-CPU2.18.1 stack. "
    f"Readout: {run['device']}, CUDA{run['cuda']}, {json.dumps(run['versions'])}, "
    "FP32, TF32disabled. Readout CUDA memory excludes frozen encoder end-to-end residency.",
    "",
    f"- Training source commit: `{run['source_commit']}`.",
    f"- Effective config SHA256: `{run['config_sha256']}`.",
    f"- Visual cache receipt SHA256: `{run['visual_cache_sha256']}`.",
    f"- Control receipt SHA256: `{run['control_receipt_sha256']}`.",
    f"- Protected prior artifacts checked unchanged: {r['preserved_artifact_count']}.",
    "",
    "Exact commands, corpus/config/checkpoint/feature hashes, timestamps, all metrics, "
    "resource records and failure history are in the companion JSON and local receipts. "
    "Weights/media/large arrays remain outside Git. Best/final checkpoints include optimizer "
    "and RNG snapshots, but this runner has no validated resume path; exact replay is inference "
    "verification, not a claim of fully reproducible training resume.",
    "",
    "## Interpretation and remaining work",
    "",
    "Interpret the predefined retention result together with every modality, uncertainty "
    "and TRAIN/validation curve. Spatial cropping was a real information restriction; "
    "this comparison tests one full-extent policy, not every possible visual representation. "
    "Do not turn a negative result into a general model-capacity ceiling. Do not infer "
    "grokking solely from later Accuracy gains while NLL/calibration deteriorates. ",
    "",
    "The broad objective remains incomplete:95–98% task quality, calibrated probabilities, "
    "native conditional/temporal/joint/OOD utility and deployable package/runtime must all "
    "be demonstrated. Current corpus has no synchronized audio/video joint decision. "
    "No further geometry/seed search, audit or product adoption is triggered automatically.",
    "",
    "Local artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-*`. "
    "The existing Teacher/QAT artifacts, original control and failed support experiment "
    "are preserved.",
    "",
]
(OUT / "report.md").write_text("\n".join(lines), encoding="utf8")
print(
    "Report generated from verified measurements; "
    "add final interpretation and CI evidence after review."
)
