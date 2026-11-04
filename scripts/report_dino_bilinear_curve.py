"""Summarize the exact-prefix DINO bilinear learning curve."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read
from tiny_omni_decision.spatial_curve import require_prefix


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    runs = {"24epochs": "dinov2-bilinear-curve-01", "8epochs": "dinov2-interaction-bilinear-01"}
    data = {
        name: dict(
            result=read(root / path / "result.json"),
            history=read(root / path / "history.json"),
            config=read(root / path / "config.json"),
            hashes={p.name: sha(p) for p in sorted((root / path).glob("*.json"))},
        )
        for name, path in runs.items()
    }
    require_prefix(data["24epochs"]["history"][:8], data["8epochs"]["history"])
    assert len(data["24epochs"]["history"]) == 24
    assert data["24epochs"]["result"]["reload_exact"]
    delta = paired(
        read(root / runs["8epochs"] / "matched-predictions.json"),
        read(root / runs["24epochs"] / "matched-predictions.json"),
    )
    out = Path("docs/pretrained_reuse/R2_DINO_BILINEAR_CURVE_RESULTS_20261012")
    write(out.with_suffix(".json"), dict(runs=data, delta=delta, prefix_exact=True))
    lines = [
        "# DINO bilinear24-epoch learning curve",
        "",
        "Fresh seed17; complete first8-epoch history matches the original exactly.",
        "Same pinned features, all8424/512 images, four10-option taxonomies,147712-parameter head.",
        "No encoder changes, new data, re-extraction, sealed audit or reporting-set inference.",
        "This is reused model-selection validation, not blind or general-image quality.",
        "",
        "| Budget | Task | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, item in data.items():
        r = item["result"]
        for task, m in {"all": r["matched"], **r["matched_taxonomies"]}.items():
            lines.append(
                f"| {name} | {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    low, high = delta["percentile_95_interval"]
    lines += [
        "",
        f"24-budget minus8-budget: {delta['difference'] * 100:+.3f}pp; "
        f"95% interval[{low * 100:+.3f},{high * 100:+.3f}]pp.",
        "512-image paired percentile bootstrap,10000 draws,seed17; selection bias remains.",
        "",
        "| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in data["24epochs"]["history"]:
        t, v = row["train"], row["validation"]
        lines.append(
            f"| {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
            f"{t['nll']:.6f} | {v['nll']:.6f} |"
        )
    r = data["24epochs"]["result"]
    lines += [
        "",
        "## Resource and artifact record",
        "",
        f"- Best epoch{r['best_epoch']}; selected{r['selected_consumed_questions']} encounters; "
        f"whole run{r['consumed_questions']}, unique33696 questions/8424 images.",
        f"- Rotated-image Accuracy{r['rotated']['accuracy']:.6f}.",
        f"- Parameters{r['total_parameters']:,}; only{r['readout_parameters']:,} trainable.",
        f"- Cache load{r['extraction_seconds']:.3f}s; "
        f"train/eval{r['training_evaluation_seconds']:.3f}s; "
        f"total{r['total_seconds']:.3f}s; peak VRAM{r['peak_vram_bytes']} bytes; local cost$0.",
        "- Best/final files separate; exact best reload; original weights/features unchanged.",
        f"- Artifacts `{(root / runs['24epochs']).as_posix()}`; all predictions/hashes retained.",
        "- Complete four-modality95% goal remains unmet.",
    ]
    lines += [
        "",
        "## Interpretation",
        "",
        "Retain this as a stronger narrow image candidate: Accuracy/NLL/Brier improve,",
        "but aggregate ECE rises0.01102→0.01784. Texture Accuracy improves while its NLL",
        "worsens0.20164→0.24176; calibration trade-offs must not be hidden.",
        "Best23 train91.98% versus validation89.26% leaves a2.72pp gap; final24 worsens",
        "both train and validation, so that step alone is not pure-overfitting evidence.",
        "No clean plateau or universal capacity limit is established, but do not extend",
        "epochs automatically. This candidate has not passed independent audit or calibration.",
        "",
        "Saved predictions show220 errors:184 count,30 texture,6 color,0 shape. Thus count",
        "accounts for83.64% of errors. Count absolute errors:328 exact,161 off-by-one,21",
        "off-by-two,2 off-by-three. These are diagnostics; exact-count Accuracy remains",
        "64.06%, and the benchmark tolerance is not relaxed. Shape100% is not image95%.",
        "Next prioritize whether retained spatial information and candidate interaction",
        "can improve exact counting, with unchanged options/splits, rather than declaring",
        "the near-misses correct or adding epochs without a distinct hypothesis.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
