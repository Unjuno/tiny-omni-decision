"""Summarize the pinned concat/bilinear interaction experiment."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    data, predictions = {}, {}
    for policy in ["concat", "bilinear"]:
        folder = root / f"dinov2-interaction-{policy}-01"
        result = read(folder / "result.json")
        assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
        assert result["reload_exact"] and result["frozen_weights_unchanged"]
        data[policy] = dict(
            result=result,
            history=read(folder / "history.json"),
            config=read(folder / "config.json"),
            hashes={p.name: sha(p) for p in sorted(folder.glob("*.json"))},
            best_bytes=(folder / "best-readout.pt").stat().st_size,
        )
        predictions[policy] = read(folder / "matched-predictions.json")
    assert data["concat"]["history"] == read(root / "dinov2-clevr4-probe-01/history.json")
    delta = paired(predictions["concat"], predictions["bilinear"])
    data["paired_bilinear_minus_concat"] = delta
    out = Path("docs/pretrained_reuse/R2_BILINEAR_RESULTS_20261012")
    write(out.with_suffix(".json"), data)
    lines = [
        "# Capacity-matched candidate interaction results",
        "",
        "Execution PASS. Concat reproduces all eight prior epoch records and the",
        "saved best tensor-file SHA exactly. Frozen CLS+patch-mean768 and question/",
        "candidate384 features; only the interaction readout differs. No feature",
        "normalization, new data, encoder tuning, or temperature selection.",
        "Concat147,713 parameters; bilinear147,712. Same seed17,8epochs,AdamW1e-3",
        "wd1e-4,batch64, question order and validation-NLL selection. Initialization",
        "parameterization differs; this is not an isolated language-removal experiment.",
        "",
        "8,424 train images/33,696 unique questions;269,568 encounters per run.",
        "512 reused selection images/2,048 questions. No reporting1595 images",
        "or sealed audit. Synthetic taxonomies do not measure generic image reasoning.",
        "",
        "| Policy | Taxonomy | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for policy in ["concat", "bilinear"]:
        r = data[policy]["result"]
        for task, m in {"all": r["matched"], **r["matched_taxonomies"]}.items():
            lines.append(
                f"| {policy} | {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    low, high = delta["percentile_95_interval"]
    lines += [
        "",
        f"Bilinear-minus-concat {delta['difference'] * 100:+.3f}pp; "
        f"paired95% interval[{low * 100:+.3f},{high * 100:+.3f}]pp.",
        "512-image paired bootstrap,10,000 draws,seed17,percentile interval;",
        "validation-selection bias remains. All probabilities retained locally.",
        "",
        "## Learning curves",
        "",
        "| Policy | Epoch | Train Acc | Validation Acc | Train NLL | Validation NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ["concat", "bilinear"]:
        for row in data[policy]["history"]:
            t, v = row["train"], row["validation"]
            lines.append(
                f"| {policy} | {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
                f"{t['nll']:.6f} | {v['nll']:.6f} |"
            )
    lines += ["", "## Artifacts and compute", ""]
    for policy in ["concat", "bilinear"]:
        r = data[policy]["result"]
        lines += [
            f"- {policy}: selected epoch{r['best_epoch']}, "
            f"selected consumption{r['selected_consumed_questions']:,}; "
            f"whole run{r['consumed_questions']:,}.",
            f"  Total{r['total_seconds']:.3f}s, training/evaluation"
            f" {r['training_evaluation_seconds']:.3f}s, peak VRAM{r['peak_vram_bytes']:,}bytes.",
            f"  Rotated-image Acc{r['rotated']['accuracy']:.6f}; "
            f"best readout{data[policy]['best_bytes']:,}bytes.",
        ]
    lines += [
        "",
        "FP32, local RTX3080Laptop16GB; cloud cost$0. Two separate successful",
        "eight-image smoke processes preceded full runs; smoke weights never reused.",
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-interaction-{policy}-01`.",
        "Exact source commit, commands, versions, base/cache hashes and result hashes",
        "in companion JSON. Best/final separate and reload exact; no exact optimizer resume.",
        "Old checkpoints, feature caches and corpora unchanged. Four-modality95–98%",
        "remains unproven; do not extrapolate a taxonomy result to product readiness.",
        "",
        "## Decision and limits",
        "",
        "Retain bilinear as the stronger measured DINO readout candidate: all four",
        "taxonomy Accuracies rise and aggregate NLL/Brier/ECE improve at essentially",
        "equal parameter count. Color/texture ECE individually worsens, so aggregate",
        "calibration improvement is not a claim that every task is better calibrated.",
        "Train89.23% versus validation87.40% at selected epoch8; NLL0.2636 versus",
        "0.2893. The endpoint is best NLL, so no clean plateau has been established.",
        "No automatic extra epochs or product promotion. Count59.77% remains weak.",
        "",
        "The existing larger VideoPrism reference remains87.89% on these IDs; the",
        "new DINO result does not establish superiority or statistical noninferiority",
        "to it. The full DINO/text/readout stack is39,594,112 parameters before",
        "quantization. Its improvement supports a readout bottleneck in this fixed",
        "image setup, not a universal frozen-feature claim or isolated causal proof",
        "about one mathematical operation.",
        "",
        "Next useful transfer test is the same interaction change on fixed video",
        "features with text-only and shuffled-scene controls. Do not infer temporal",
        "reasoning from these static image results or discard the controls.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
