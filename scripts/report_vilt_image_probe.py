"""Compare completed ViLT/DINO probes on the same observed image validation IDs."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    data, predictions = {}, {}
    for name, path in [("vilt", "vilt-image-probe-01"), ("dino", "dinov2-interaction-bilinear-01")]:
        folder = root / path
        result = read(folder / "result.json")
        assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
        assert result["reload_exact"] and result["frozen_weights_unchanged"]
        data[name] = dict(
            result=result,
            history=read(folder / "history.json"),
            config=read(folder / "config.json"),
            hashes={p.name: sha(p) for p in sorted(folder.glob("*.json"))},
        )
        predictions[name] = read(folder / "matched-predictions.json")
        assert len(data[name]["history"]) == 8
    delta = paired(predictions["dino"], predictions["vilt"])
    data["paired_vilt_minus_dino"] = delta
    out = Path("docs/pretrained_reuse/R2_VILT_IMAGE_RESULTS_20261012")
    write(out.with_suffix(".json"), data)
    lines = [
        "# Frozen ViLT conditioned image readout",
        "",
        "Both use8424 training images/33696 questions and512 validation images/2048 questions.",
        "All four10-option taxonomies retained, including9 options absent from ViLT's closed head.",
        "Only147712-parameter bilinear readout trained; encoders frozen, seed17,8epochs,",
        "AdamW1e-3/weight-decay1e-4,batch64,earliest minimum validation NLL selection.",
        "This compares complete representations/preprocessing, not fusion alone:",
        "ViLT uses question-conditioned published384×512 input; DINO uses224×224 image features.",
        "No prompt search, new data, sealed audit or1595-image reporting set evaluation.",
        "",
        "| Model | Taxonomy | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, item in data.items():
        if name not in ["vilt", "dino"]:
            continue
        r = item["result"]
        for task, m in {"all": r["matched"], **r["matched_taxonomies"]}.items():
            lines.append(
                f"| {name} | {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    low, high = delta["percentile_95_interval"]
    lines += [
        "",
        f"ViLT-minus-DINO {delta['difference'] * 100:+.3f}pp, "
        f"95% paired interval[{low * 100:+.3f},{high * 100:+.3f}]pp.",
        "512-image paired percentile bootstrap,10000 draws,seed17; selection bias remains.",
        "",
        "## Learning curve",
        "",
        "| Epoch | Train Acc | Validation Acc | Train NLL | Validation NLL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in data["vilt"]["history"]:
        t, v = row["train"], row["validation"]
        lines.append(
            f"| {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
            f"{t['nll']:.6f} | {v['nll']:.6f} |"
        )
    r = data["vilt"]["result"]
    lines += [
        "",
        "## Resource/integrity results",
        "",
        f"- Best epoch{r['best_epoch']}; selected{r['selected_consumed_questions']} encounters, "
        f"whole run{r['consumed_questions']};33696 unique questions and8424 unique images.",
        f"- Rotated-image Accuracy{r['rotated']['accuracy']:.6f}, NLL{r['rotated']['nll']:.6f}.",
        f"- Total resident parameters{r['total_parameters']:,}, including unused VQA classifier.",
        f"- Extraction{r['extraction_seconds']:.3f}s; "
        f"train/eval{r['training_evaluation_seconds']:.3f}s; "
        f"total{r['total_seconds']:.3f}s; peak VRAM{r['peak_vram_bytes']}bytes; cost$0.",
        "- Best/final checkpoints separate; exact best reload; original weights unchanged.",
        "- Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/vilt-image-probe-01`.",
        "  All probabilities, feature caches, configs and hashes retained outside Git.",
        "- Execution smoke completed separately before the full run.",
        "- Fixed taxonomy performance is not general image/video/omni decision quality.",
        "  ViLT feature extraction requires a new joint forward for each new question.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
