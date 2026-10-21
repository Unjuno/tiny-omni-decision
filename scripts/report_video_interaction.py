"""Summarize fixed pooled-video interaction controls; no model inference."""

import hashlib
import json
from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    folder = root / "video-interaction-01"
    result, config = (read(folder / f"{name}.json") for name in ["result", "config"])
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert read(folder / "no_video/predictions.json") == read(
        folder / "no_video/shuffled-predictions.json"
    )
    histories = {p: read(folder / p / "history.json") for p in result["conditions"]}
    assert all(len(h) == 24 for h in histories.values())
    ids = config.pop("ids")
    config["id_counts"] = {k: len(v) for k, v in ids.items()}
    config["ordered_ids_sha256"] = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    out = Path("docs/pretrained_reuse/R2_VIDEO_INTERACTION_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=config,
            histories=histories,
            hashes={str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*.json"))},
            smoke02=read(root / "video-interaction-smoke-02/result.json"),
            invalid_smoke01_result_sha256=sha(root / "video-interaction-smoke-01/result.json"),
        ),
    )
    lines = [
        "# Pooled-video interaction comparison",
        "",
        "Execution PASS for all three24-epoch runs; quality gate remains unmet.",
        f"Source `{config['source_commit']}`. Separate fresh seed17 runs, identical",
        "sampling order, fixed weighted CE and validation macro-NLL selector.",
        "Bilinear and zero-video controls have byte-identical initial states.",
        "3794 unique choices/256 training scenes;3014 choices/200 validation scenes.",
        "91,056 encounters per condition;87,262 repeats, not additional unique data.",
        "Validation was previously observed and used for checkpoint selection.",
        "No separate control reporting set or sealed audit was opened.",
        "",
        "## Selected validation metrics",
        "",
        "| Policy/input | Task | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for policy, condition in result["conditions"].items():
        for variant in ["matched", "shuffled"]:
            metrics = condition[variant]
            for task, m in {
                "macro": metrics["macro_question_type"],
                **metrics["by_question_type"],
            }.items():
                lines.append(
                    f"| {policy}/{variant} | {task} | {m['accuracy']:.6f} | "
                    f"{m['nll']:.6f} | {m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
                )
    lines += ["", "## Paired macro-Accuracy uncertainty", ""]
    for name, value in result["comparisons"].items():
        low, high = value["percentile_95_interval"]
        lines.append(
            f"- {name}: {value['difference'] * 100:+.3f}pp; "
            f"95% interval[{low * 100:+.3f},{high * 100:+.3f}]pp."
        )
    lines += [
        "",
        "200-scene paired bootstrap,10,000 draws,seed17,percentile intervals.",
        "No simultaneous correction; these do not remove checkpoint-selection bias.",
        "Zero-video matched/shuffled predictions are identical as required.",
        "",
        "## Learning and checkpoints",
        "",
        "| Policy | Epoch | Train macro Acc | Val macro Acc | Train macro NLL | Val macro NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, history in histories.items():
        for row in history:
            t, v = row["train"]["macro_question_type"], row["validation"]["macro_question_type"]
            lines.append(
                f"| {policy} | {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
                f"{t['nll']:.6f} | {v['nll']:.6f} |"
            )
    lines += ["", "## Resources and integrity", ""]
    for policy, r in result["conditions"].items():
        lines.append(
            f"- {policy}: best epoch{r['best_epoch']}, "
            f"selected{r['selected_consumed']:,}/whole{r['consumed']:,} encounters, "
            f"{r['parameters']:,} trainable parameters,{r['seconds']:.3f}s."
        )
    lines += [
        f"- Total{result['seconds']:.3f}s; peak allocated VRAM{result['peak_vram_bytes']:,}bytes.",
        "- Local RTX3080Laptop16GB,FP32; cloud cost$0. Frozen text weights/video cache",
        "  hash checked before and after; same underlying scene/media/content gates.",
        "- Full video/text/bilinear deployment components total132,050,176 parameters",
        "  before quantization (114,365,184+17,389,824+295,168). No encoder loaded for",
        "  video extraction here; cached features used. This is not all-modality quality.",
        "- Best and final weights saved separately; best reload logits exactly match.",
        "- Old files untouched. Artifacts: "
        "`C:/CodexArtifacts/pretrained-reuse-r2/video-interaction-01`.",
        "  All probabilities, full config and IDs remain local; hashes/summaries in JSON.",
        "",
        "## Smoke failure retained",
        "",
        "smoke-01 completed compute but its tiny-scene bootstrap generated undefined",
        "replicates and NaN intervals; its PASS_EXECUTION_ONLY flag does not validate",
        "those intervals. Exclude them from quality conclusions. Preserve the file.",
        "The corrected evaluator omits quality bootstrap for smoke and rejects any",
        "missing-type replicate in full evaluation. Regression tests and smoke-02",
        "passed before the full run. No old result was overwritten or silently repaired.",
        "",
        "Four-modality95–98% remains unmet. Static-image readout gains do not",
        "establish pooled-video temporal reasoning. No automatic extra training.",
        "",
        "## Decision and bottleneck evidence",
        "",
        "Do not adopt bilinear for this pooled-video setup: macro Accuracy gain",
        "over concat is only0.220pp with an interval spanning zero, and selected",
        "NLL/Brier/ECE are worse. The no-video comparison likewise remains uncertain.",
        "Matched-minus-shuffled+1.829pp is suggestive but its interval includes zero;",
        "neither a reliable scene benefit nor complete absence of video information",
        "is established. All three question types remain far below95%.",
        "",
        "Bilinear train macro NLL falls0.6781→0.6536 from epoch12 to24 while validation",
        "NLL rises0.6934→0.7075. This is explicit overfitting evidence; best12 is",
        "retained, not the final24. Fixed text preprocessing truncated zero questions",
        "(maximum44 train/45 validation tokens), so truncation does not explain this run.",
        "",
        "Further pooled-readout tuning or more epochs is not justified here. Before",
        "another quality run, inspect reusable spatiotemporal tokens and their real",
        "memory/runtime cost at the same frame count. This is a next diagnostic, not",
        "a claim that unpooled tokens will solve video reasoning or a base capacity limit.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
