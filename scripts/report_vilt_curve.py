"""Report the fixed ViLT learning curve and paired baseline comparisons."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read
from tiny_omni_decision.spatial_curve import require_prefix


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    runs = {
        "vilt24": "vilt-image-curve-01",
        "vilt8": "vilt-image-probe-01",
        "dino8": "dinov2-interaction-bilinear-01",
    }
    data = {
        k: {
            "result": read(root / p / "result.json"),
            "history": read(root / p / "history.json"),
            "config": read(root / p / "config.json"),
            "hashes": {x.name: sha(x) for x in sorted((root / p).glob("*.json"))},
        }
        for k, p in runs.items()
    }
    require_prefix(data["vilt24"]["history"][:8], data["vilt8"]["history"])
    assert len(data["vilt24"]["history"]) == 24
    assert data["vilt24"]["result"]["reload_exact"]
    assert data["vilt24"]["result"]["frozen_weights_unchanged"]
    pred = {k: read(root / p / "matched-predictions.json") for k, p in runs.items()}
    comparisons = {k: paired(pred[k], pred["vilt24"]) for k in ["vilt8", "dino8"]}
    out = Path("docs/pretrained_reuse/R2_VILT_CURVE_RESULTS_20261012")
    write(out.with_suffix(".json"), dict(runs=data, comparisons=comparisons, prefix_exact=True))
    lines = [
        "# ViLT fixed24-epoch learning curve",
        "",
        "Fresh seed17; the entire first8-epoch history reproduces exactly.",
        "No resume or re-extraction. Pinned cache/config/history hashes verified.",
        "8424 train images/33696 questions;512 observed validation images/2048 questions.",
        "All four10-option taxonomies remain intact; no sealed audit or reporting set read.",
        "",
        "| Run | Task | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, item in data.items():
        r = item["result"]
        for task, m in {"all": r["matched"], **r["matched_taxonomies"]}.items():
            lines.append(
                f"| {name} | {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    for name, comp in comparisons.items():
        low, high = comp["percentile_95_interval"]
        lines += [
            "",
            f"ViLT24 minus {name}: {comp['difference'] * 100:+.3f}pp, "
            f"95% interval[{low * 100:+.3f},{high * 100:+.3f}]pp.",
        ]
    lines += [
        "",
        "512-image paired percentile bootstrap,10000 draws,seed17; selection bias remains.",
        "",
        "| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in data["vilt24"]["history"]:
        t, v = row["train"], row["validation"]
        lines.append(
            f"| {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
            f"{t['nll']:.6f} | {v['nll']:.6f} |"
        )
    r = data["vilt24"]["result"]
    lines += [
        "",
        "## Compute and artifacts",
        "",
        f"- Best epoch{r['best_epoch']}, selected{r['selected_consumed_questions']} encounters; "
        f"whole run{r['consumed_questions']} (unique33696).",
        f"- Rotated-image Accuracy{r['rotated']['accuracy']:.6f}.",
        f"- Cache load{r['extraction_seconds']:.3f}s, train/eval "
        f"{r['training_evaluation_seconds']:.3f}s, total{r['total_seconds']:.3f}s, "
        f"peak allocated VRAM{r['peak_vram_bytes']} bytes; cost$0.",
        "- Parameter total remains135,126,073; only147,712 readout parameters trained.",
        "- Best/final files separate; reload exact. No old artifacts changed.",
        "- Directory `C:/CodexArtifacts/pretrained-reuse-r2/vilt-image-curve-01`.",
        "- Original8-epoch config representation string was a DINO scaffold typo;",
        "  executed code/cache used ViLT pooler_output. New config corrects the label;",
        "  original metadata is retained. No result is silently replaced.",
        "- These are narrow taxonomy results, not general image or all-modality95%.",
    ]
    lines += [
        "",
        "## Interpretation and next priority",
        "",
        "Extra optimization improves ViLT Accuracy/NLL/Brier on this observed validation",
        "set, but ECE rises0.01879→0.02242. Selected23 is retained separately from final24.",
        "ViLT remains2.93pp below the much smaller DINO candidate; do not replace it.",
        "Texture81.64% and count56.84% remain well below95% even though color reaches100%.",
        "A single taxonomy's100% must not be presented as image or omni success.",
        "",
        "At23 train85.24%/NLL0.3707 versus validation84.47%/NLL0.3995; at24 both worsen.",
        "The curve includes optimization fluctuation and further late improvement; it",
        "does not establish a clean plateau or a model-family capacity limit. Nevertheless",
        "the relative quality/size result does not justify automatically extending ViLT.",
        "Prioritize measuring the stronger39.59M DINO/bilinear candidate's matched learning",
        "curve before further work on the135.13M ViLT stack. No additional run is implied",
        "by this report; its data, initialization and prefix gates must be fixed separately.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
