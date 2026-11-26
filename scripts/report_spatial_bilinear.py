"""Report the preregistered matched spatial/bilinear experiment."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "dinov2-spatial-bilinear-01"
    result = read(run / "result.json")
    config = read(run / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert config["epochs"] == 24 and config["readout"] == "bilinear"
    histories = {k: read(run / k / "history.json") for k in ["global", "spatial"]}
    predictions = {k: read(run / k / "matched-predictions.json") for k in histories}
    reference = read(root / "dinov2-bilinear-curve-01/matched-predictions.json")
    for key, history in histories.items():
        assert len(history) == 24
        r = result["conditions"][key]
        assert r["reload_exact"] and r["consumed"] == 808704
        assert r["best_epoch"] == min(history, key=lambda x: x["validation"]["nll"])["epoch"]
    a, b = result["conditions"].values()
    assert a["initial_tensor_hash"] == b["initial_tensor_hash"]
    assert a["order_sha256"] == b["order_sha256"]
    comparisons = {"spatial_minus_global": paired(predictions["global"], predictions["spatial"])}
    comparisons.update({k + "_minus_small": paired(reference, p) for k, p in predictions.items()})
    out = Path("docs/pretrained_reuse/R2_SPATIAL_BILINEAR_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            config=config,
            result=result,
            histories=histories,
            paired_comparisons=comparisons,
            hashes={str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*.json"))},
            smoke=read(root / "dinov2-spatial-bilinear-smoke-01/result.json"),
        ),
    )
    lines = [
        "# Spatial information with a bilinear candidate readout",
        "",
        "Execution PASS: two fresh seed17 runs completed24epochs, with identical initial",
        "weights and shuffled question order. Both best reloads match exactly. All",
        "frozen weights and pinned cache hashes retained; global-feature parity passed.",
        "Source commit `" + config["source_commit"] + "`; exact argv/config in companion JSON.",
        "",
        "Each condition uses40,331,392 total parameters,884,992 trainable. Same width",
        "does not imply equal effective input rank: global repeats the mean spatial cell.",
        "Train8,424 unique images/33,696 questions; validation512 images/2,048 questions.",
        "Per run808,704 encounters,775,008 repeats. No new data or sealed audit.",
        "Best selected by earliest minimum validation NLL; final saved separately.",
        "",
        "| Condition | Best epoch | Selected encounters | Acc | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in result["conditions"].items():
        m = r["matched"]
        lines.append(
            f"| {k} | {r['best_epoch']} | {r['selected_consumed']} | "
            f"{m['accuracy']:.6f} | {m['nll']:.6f} | {m['brier']:.6f} | "
            f"{m['ece_15_bins']:.6f} |"
        )
    lines += [
        "",
        "## Per-task selected metrics",
        "",
        "| Condition | Task | Acc | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for k, r in result["conditions"].items():
        for task, m in r["matched_taxonomies"].items():
            lines.append(
                f"| {k} | {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += [
        "",
        "## Complete learning curves",
        "",
        "| Condition | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, history in histories.items():
        for row in history:
            t, v = row["train"], row["validation"]
            lines.append(
                f"| {k} | {row['epoch']} | {t['accuracy']:.6f} | "
                f"{v['accuracy']:.6f} | {t['nll']:.6f} | {v['nll']:.6f} |"
            )
    lines += ["", "## Paired comparisons", ""]
    for name, c in comparisons.items():
        lo, hi = c["percentile_95_interval"]
        lines.append(
            f"- {name}: {100 * c['difference']:+.3f}pp,95% percentile interval"
            f"[{100 * lo:+.3f},{100 * hi:+.3f}]pp."
        )
    lines += [
        "",
        "Paired512-image bootstrap,10,000 draws,seed17. Validation has been reused",
        "for model selection; these intervals do not remove selection bias.",
        "",
        "## Compute and controls",
        "",
        f"Total{result['total_seconds']:.3f}s; peak allocated VRAM"
        f"{result['peak_vram_bytes']:,}bytes; {result['gpu']}; cloud cost$0.",
        "No encoder extraction; existing feature cache only. Rotated-image Accuracy:",
    ]
    for k, r in result["conditions"].items():
        lines.append(f"- {k}: {r['rotated']['accuracy']:.6f}.")
    lines += [
        "",
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-spatial-bilinear-01`.",
        "Weights stay outside Git. Full checkpoint hashes, gradients and task curves",
        "are recorded in JSON. This narrow image diagnostic does not establish",
        "general image reasoning or four-modality95–98% quality.",
    ]
    lines += [
        "",
        "## Interpretation and next decision",
        "",
        "Spatial retention improves the matched-capacity global control by1.56pp",
        "and exact counting by7.03pp (63.48% to70.51%). This supports useful spatial",
        "information for this readout on reused development validation. However,",
        "the gain over the stronger small bilinear reference (89.26%) is only0.78pp",
        "with an interval crossing zero. Keep both artifacts; no product replacement",
        "or robust superiority claim. The new head is737,280 parameters larger.",
        "",
        "Against the small reference, color98.83% to97.46%, texture94.14% to92.38%,",
        "shape100% to99.80% decrease while counting64.06% to70.51% improves.",
        "NLL/Brier improve slightly overall, but ECE worsens from0.017840 to0.022108.",
        "This is a measured trade-off, not universally better calibration or quality.",
        "",
        "Spatial epoch16 has train Accuracy93.11% versus90.04% validation and",
        "NLL0.168861 versus0.253458. At22 train NLL improves to0.148432 while",
        "validation worsens to0.280281: an overfitting warning. Epoch24 is not best.",
        "Preserve the epoch16 checkpoint; do not extend this fixed run automatically.",
        "The next design should test generalizable spatial/object readout, with a",
        "training-only fit diagnostic before further selector reuse. A linear visual",
        "projection conditioned by candidates does not explicitly model object relations.",
        "",
        "Verification: Ruff passes;394 CPU tests pass,1 skips; all CI manifest/catalog",
        "validation commands pass locally. Final commit CI is checked after push.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
