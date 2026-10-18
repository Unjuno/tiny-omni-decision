"""Report the preregistered matched spatial/bilinear experiment."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "dinov2-mirror-control-01"
    result = read(run / "result.json")
    config = read(run / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert config["epochs"] == 24 and config["readout"] == "bilinear"
    histories = {k: read(run / k / "history.json") for k in ["original", "augmented"]}
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
    comparisons = {
        "augmented_minus_original": paired(predictions["original"], predictions["augmented"])
    }
    comparisons.update({k + "_minus_small": paired(reference, p) for k, p in predictions.items()})
    out = Path("docs/pretrained_reuse/R2_MIRROR_CONTROL_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            config=config,
            result=result,
            histories=histories,
            paired_comparisons=comparisons,
            hashes={str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*.json"))},
            smoke=read(root / "dinov2-mirror-smoke-01/result.json"),
        ),
    )
    lines = [
        "# Matched physical-mirror augmentation comparison",
        "",
        "Execution PASS: two fresh seed17 runs completed24epochs, with identical initial",
        "weights and shuffled question order. Both best reloads match exactly. All",
        "frozen weights and pinned cache hashes retained; global-feature parity passed.",
        "Source commit `" + config["source_commit"] + "`; exact argv/config in companion JSON.",
        "",
        "Each condition uses40,331,392 total parameters,884,992 trainable. Same width",
        "and architecture; only training observation view policy differs.",
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
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-mirror-control-01`.",
        "Weights stay outside Git. Full checkpoint hashes, gradients and task curves",
        "are recorded in JSON. This narrow image diagnostic does not establish",
        "general image reasoning or four-modality95–98% quality.",
    ]
    lines += [
        "",
        "## Scope and safeguards",
        "",
        "Original-only history matches all24 saved epochs exactly; verified in training.",
        "Augmentation uses one independent seed17 mask per image/epoch, shared by4tasks.",
        "Train metrics use original images for both models, not changing training views.",
        "Realized mirrored counts are in JSON. Unique original assets remain8424.",
        "No inference ensemble, calibration, new validation images or audit. Best and final",
        "checkpoints are separate. This is one seed on reused selector data.",
    ]
    lines += [
        "",
        "## Decision",
        "",
        "No convincing Accuracy improvement:90.04% to90.19% is3 additional correct",
        "answers out of2048, with paired interval spanning zero. Aggregate NLL is",
        "worse (0.253458 to0.258652), although Brier/ECE improve. Counting Accuracy",
        "decreases70.51% to69.53% (5/512 fewer), despite improved counting NLL/Brier/ECE.",
        "Texture NLL worsens0.228276 to0.283977. Keep both records, retain the prior",
        "lower-NLL reference; do not adopt mirror augmentation as a proven fix.",
        "",
        "The small physical-mirror robustness diagnostic justified this controlled",
        "experiment but did not predict a validation gain. This budget does not support",
        "augmentation alone solving counting. Do not extend epochs or tune reflection",
        "probabilities against this selector. Remaining spatial readout/object relation",
        "limitations need a different, separately specified hypothesis.",
        "",
        "Ruff passes;395 CPU tests pass,1skip. Existing checkpoints/corpora preserved.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
