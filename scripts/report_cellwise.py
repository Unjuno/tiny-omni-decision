"""Report the fixed shared-cell readout comparison."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "dinov2-cellwise-01"
    result = read(run / "result.json")
    config = read(run / "config.json")
    history = read(run / "spatial/history.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert len(history) == 24 and config["readout"] == "cellwise"
    selected = result["conditions"]["spatial"]
    assert selected["reload_exact"]
    assert selected["best_epoch"] == min(history, key=lambda x: x["validation"]["nll"])["epoch"]
    predictions = read(run / "spatial/matched-predictions.json")
    comparisons = {
        name: paired(read(root / path), predictions)
        for name, path in {
            "minus_spatial_bilinear": "dinov2-spatial-bilinear-01/spatial/matched-predictions.json",
            "minus_small_bilinear": "dinov2-bilinear-curve-01/matched-predictions.json",
        }.items()
    }
    out = Path("docs/pretrained_reuse/R2_CELLWISE_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=config,
            history=history,
            comparisons=comparisons,
            hashes={str(p.relative_to(run)): sha(p) for p in run.rglob("*.json")},
        ),
    )
    lines = [
        "# Shared nonlinear cell readout: fixed24-epoch comparison",
        "",
        "Execution PASS, exact best reload; all pinned feature/model hashes retained.",
        f"Source `{config['source_commit']}`; exact argv/config/hashes in companion JSON.",
        "Frozen DINO/MiniLM plus180736-parameter head =39,627,136 total parameters.",
        "Train8424 unique images/33696 questions, validation512/2048, all options.",
        f"Best epoch{selected['best_epoch']}; "
        f"selected encounters{selected['selected_consumed']}; total808704.",
        "Final checkpoint saved separately. Repeated encounters are not independent data.",
        "No augmentation, new validation, audit, calibration or ensemble.",
        "",
        "| Task | Acc | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, m in {"all": selected["matched"], **selected["matched_taxonomies"]}.items():
        lines.append(
            f"| {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
            f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
        )
    lines += [
        "",
        "## Learning curve",
        "",
        "| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in history:
        t, v = row["train"], row["validation"]
        lines.append(
            f"| {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
            f"{t['nll']:.6f} | {v['nll']:.6f} |"
        )
    lines += ["", "## Paired comparisons", ""]
    for name, c in comparisons.items():
        lo, hi = c["percentile_95_interval"]
        lines.append(
            f"- {name}: {c['difference'] * 100:+.3f}pp;95%[{lo * 100:+.3f},{hi * 100:+.3f}]pp."
        )
    lines += [
        "",
        "Paired512-image percentile bootstrap,10000 draws,seed17. These reused",
        "selector-data intervals do not remove checkpoint/model-selection bias.",
        "This jointly changes nonlinearity, weight sharing and parameter count;",
        "do not attribute results to only one factor or claim object-level reasoning.",
        "",
        f"Total{result['total_seconds']:.3f}s, "
        f"peak allocated VRAM{result['peak_vram_bytes']:,}bytes, cloud$0.",
        f"Rotated-image Accuracy{selected['rotated']['accuracy']:.6f}.",
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-cellwise-01`.",
        "No weights/features in Git; old artifacts preserved. "
        "General four-modality95% remains unmet.",
    ]
    lines += [
        "",
        "## Interpretation and decision",
        "",
        "The predeclared NLL selector chooses epoch15,89.79% Accuracy, not the91.06%",
        "Accuracy maximum at22. Do not switch selection rules after viewing this curve.",
        "Against spatial bilinear90.04%, the Accuracy difference is uncertain and NLL",
        "is essentially similar but slightly worse (0.253962 vs0.253458). The head has",
        "704256 fewer parameters, but exact counting drops70.51% to66.41%. Texture",
        "improves92.38% to94.53%; this is a trade-off, not a counting fix.",
        "",
        "Train NLL falls0.190821 at15 to0.123950 at24 while validation NLL rises",
        "0.253962 to0.286026. This is evidence of overfitting after the selected point.",
        "Retain the smaller-head artifact for cost comparison, without replacing the",
        "counting reference or claiming robust superiority. No automatic training extension.",
        "The result does not establish a frozen encoder ceiling: this head still averages",
        "contextual cells rather than detecting or relating individual objects.",
        "",
        "Ruff passes;396 CPU tests pass,1skip. Full best/final weights stay local.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
