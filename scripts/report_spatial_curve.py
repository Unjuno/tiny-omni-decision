"""Summarize the fixed 24-epoch spatial/global run and image-group uncertainty."""

import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from tiny_omni_decision.spatial_curve import require_prefix


def read(path):
    return json.loads(path.read_text())


def paired(first, second):
    groups = {}
    for a, b in zip(first, second, strict=True):
        if any(a[k] != b[k] for k in ["id", "image_id", "target", "options"]):
            raise ValueError("predictions differ in identity")
        groups.setdefault(a["image_id"], []).append(
            int(b["prediction"] == b["target"]) - int(a["prediction"] == a["target"])
        )
    assert len(groups) == 512 and all(len(v) == 4 for v in groups.values())
    delta = np.array([np.mean(v) for v in groups.values()])
    draws = delta[np.random.default_rng(17).integers(512, size=(10000, 512))].mean(1)
    return dict(
        difference=float(delta.mean()),
        percentile_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
        groups=512,
        replicates=10000,
        seed=17,
    )


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "dinov2-spatial-curve-01"
    parent = root / "dinov2-spatial-control-02"
    result = read(run / "result.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    config = read(run / "config.json")
    assert config["epochs"] == 24
    histories, predictions, comparisons = {}, {}, {}
    for condition in ["global", "spatial"]:
        histories[condition] = read(run / condition / "history.json")
        assert len(histories[condition]) == 24
        require_prefix(histories[condition][:8], read(parent / condition / "history.json"))
        predictions[condition] = read(run / condition / "matched-predictions.json")
        comparisons[condition + "_selected_minus_parent8"] = paired(
            read(parent / condition / "matched-predictions.json"), predictions[condition]
        )
    comparisons["spatial_minus_global_selected"] = paired(
        predictions["global"], predictions["spatial"]
    )
    out = Path("docs/pretrained_reuse/R2_SPATIAL_CURVE_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=config,
            histories=histories,
            paired_comparisons=comparisons,
            hashes={str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*.json"))},
            exact_eight_epoch_prefix=True,
            smoke=read(root / "dinov2-spatial-curve-smoke-01/result.json"),
        ),
    )
    lines = [
        "# Fixed spatial/global 24-epoch curve",
        "",
        "Execution PASS. Both fresh seed17 runs exactly reproduce their entire",
        "eight-epoch prior history before extending to24. No optimizer resume.",
        "Only the884,993-parameter readout trains; the deployment stack remains",
        "40,331,393 parameters including frozen DINOv2-small and MiniLM.",
        "Cached image features are byte-hashed and reused, with original global",
        "feature parity <=1e-5. No encoder update, new data or input normalization change.",
        f"Source commit `{config['source_commit']}`; full argv/hashes in companion JSON.",
        "",
        "Train8,424 unique images/33,696 unique questions; same512 selector images",
        "and2,048 questions. Per condition808,704 consumed questions at epoch24,",
        "Repeated encounters are not new data: exact repeats =808,704−33,696=775,008.",
        "No1595-image reporting subset or sealed audit opened. This is reused",
        "development validation, not blind image quality or general image reasoning.",
        "",
        "## Fixed learning curve",
        "",
        "| Condition | Epoch | Train Acc | Validation Acc | Train NLL | Validation NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, history in histories.items():
        for row in history:
            t, v = row["train"], row["validation"]
            lines.append(
                f"| {condition} | {row['epoch']} | {t['accuracy']:.6f} | "
                f"{v['accuracy']:.6f} | {t['nll']:.6f} | {v['nll']:.6f} |"
            )
    lines += ["", "## Selected checkpoints", ""]
    for condition, r in result["conditions"].items():
        lines += [
            f"### {condition}",
            "",
            f"Best epoch{r['best_epoch']}, selected consumption{r['selected_consumed']:,}; "
            f"whole run{r['consumed']:,}. Earliest minimum validation NLL.",
            "Final checkpoint retained separately; selected reload logits match exactly.",
            "",
            "| Taxonomy | Accuracy | NLL | Brier | ECE |",
            "|---|---:|---:|---:|---:|",
        ]
        for task, m in {"all": r["matched"], **r["matched_taxonomies"]}.items():
            lines.append(
                f"| {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
        lines += ["", f"Rotated-image Accuracy: {r['rotated']['accuracy']:.6f}.", ""]
    lines += ["## Paired uncertainty", ""]
    for name, c in comparisons.items():
        low, high = c["percentile_95_interval"]
        lines.append(
            f"- {name}: {c['difference'] * 100:+.3f}pp; "
            f"percentile95%[{low * 100:+.3f},{high * 100:+.3f}]pp."
        )
    lines += [
        "",
        "512-image paired bootstrap,10,000 draws,seed17; no simultaneous correction.",
        "These intervals do not remove validation selection bias.",
        "",
        "## Interpretation and next decision",
        "",
        "More optimization improved both selected results versus epoch8. Eight",
        "epochs alone was insufficient evidence for a plateau. However, the spatial",
        "gain over the capacity-matched global head remains uncertain (interval",
        "includes zero). Both trail the earlier147,713-parameter head's85.06% on",
        "these same validation images. Keep that smaller DINO reference; do not",
        "adopt this larger head or automatically extend the budget again.",
        "The strongest retained VideoPrism image reference is87.89% on these IDs.",
        "",
        "Counting remains weak:41.80% global,46.88% spatial versus53.71% for the",
        "earlier small DINO head. Color/shape above95% do not meet all-image95%.",
        "Spatial best NLL occurs at19 rather than24; later train fit improves",
        "without better validation NLL. This is an overfitting/optimization warning,",
        "not a proof of a backbone limit. Global epoch23 degrades on both train",
        "and validation, then recovers: pure generalization overfit does not explain",
        "every fluctuation. Read all curves and preserve these failed configurations.",
        "",
        "A next spatial design must address representation/readout fit rather than",
        "claiming more cell features necessarily help. No full encoder fine-tuning",
        "or new data ingestion is justified by this diagnostic alone.",
        "",
        "## Compute and preservation",
        "",
        f"- Total{result['total_seconds']:.3f}s, cache loading "
        f"{result['extraction_seconds']:.3f}s; no new encoder extraction.",
        f"- Peak allocated VRAM{result['peak_vram_bytes']:,}bytes, "
        f"GPU{result['gpu']},cloud cost$0.",
        "- Local artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-spatial-curve-01`.",
        "  Weights stay outside Git. Original caches, checkpoints and corpora retained.",
        "- Hashes, task curves, gradient norms and source configuration in JSON.",
        "- No automatic extension beyond24 or product adoption from this validation.",
        "  Four-modality95–98% remains a separate unmet goal.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
