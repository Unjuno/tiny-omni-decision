"""Report the preregistered frozen shared-text image comparison."""

from pathlib import Path

from scripts.compare_numeric_embedding_predictions import compare
from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "shared-text-image-01"
    result = read(run / "result.json")
    config = read(run / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert config["epochs"] == 24 and config["readout"] == "bilinear"
    histories = {k: read(run / k / "history.json") for k in ["minilm", "shared"]}
    predictions = {k: read(run / k / "matched-predictions.json") for k in histories}
    for key, history in histories.items():
        assert len(history) == 24
        r = result["conditions"][key]
        for label in ("best", "final"):
            assert sha(run / key / f"{label}.pt") == r[f"{label}_sha256"]
        assert r["reload_exact"] and r["consumed"] == 808704
        assert r["best_epoch"] == min(history, key=lambda x: x["validation"]["nll"])["epoch"]
    a, b = result["conditions"].values()
    assert a["initial_tensor_hash"] == b["initial_tensor_hash"]
    assert a["order_sha256"] == b["order_sha256"]
    comparisons = {"shared_minus_minilm": compare(predictions["minilm"], predictions["shared"])}
    for task in result["conditions"]["minilm"]["matched_taxonomies"]:
        comparisons[task] = compare(
            [r for r in predictions["minilm"] if r["taxonomy"] == task],
            [r for r in predictions["shared"] if r["taxonomy"] == task],
        )
    out = Path("docs/pretrained_reuse/R2_SHARED_TEXT_IMAGE_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            config=config,
            result=result,
            histories=histories,
            paired_comparisons=comparisons,
            hashes={str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*.json"))},
            smoke=read(root / "shared-text-image-gpu-smoke-01/result.json"),
        ),
    )
    lines = [
        "# Frozen shared-text image readout comparison",
        "",
        "Execution PASS: two fresh seed17 runs completed24epochs, with identical initial",
        "weights and shuffled question order. Both best reloads match exactly. All",
        "frozen weights and pinned cache hashes retained; global-feature parity passed.",
        "Source commit `" + config["source_commit"] + "`; exact argv/config in companion JSON.",
        "",
        "Both use 884,992 trainable head parameters. Image-only totals: MiniLM",
        "40,331,392 versus shared Cross-Encoder 45,655,169. Combined-system savings",
        "require reusing the SAME resident Cross-Encoder for text Decision tasks.",
        "This experiment does not implement or validate that entire combined runtime.",
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
        for metric, values in c["metrics"].items():
            lo, hi = values["percentile_95_interval"]
            lines.append(
                f"- {name}, {metric}: {values['difference']:+.6f}; "
                f"95% percentile interval [{lo:+.6f}, {hi:+.6f}]."
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
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/shared-text-image-01`.",
        "Weights stay outside Git. Full checkpoint hashes, gradients and task curves",
        "are recorded in JSON. This narrow image diagnostic does not establish",
        "general image reasoning or four-modality95–98% quality.",
    ]
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "No automatic product adoption. Report per-task trade-offs, including calibration",
        "and exact counting. A wide interval crossing zero is not proof of equivalence.",
        "The same development images selected checkpoints; intervals do not establish",
        "blind generalization or remove repeated-testing/model-selection bias.",
        "Native video, general questions, simultaneous fusion and all-modality95% remain open.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
