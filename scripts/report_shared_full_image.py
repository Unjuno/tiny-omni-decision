"""Verify the fixed whole-image global run against its cropped reference."""

import json
from pathlib import Path

import numpy as np

from scripts.report_shared_spatial_images import metrics
from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.paired_image_comparison import paired_image_metrics


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "shared-full-image-global-01"
    reference = root / "shared-spatial-image-readout-01/global"
    cache = root / "videoprism-full-image-cache-01"
    config, result = read(run / "config.json"), read(run / "result.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert set(result["conditions"]) == {"global"} and config["epochs"] == 24
    new, old = result["conditions"]["global"], read(reference / "result.json")
    assert sha(reference / "result.json") == config["cropped_reference_sha256"]
    assert config["option_embeddings_sha256"] == read(
        reference.parent / "config.json")["option_embeddings_sha256"]
    assert new["initial_tensor_hash"] == old["initial_tensor_hash"]
    assert new["order_sha256"] == old["order_sha256"]
    assert new["consumed"] == old["consumed"] == 808704
    for name, value in config["cache_hashes"].items():
        assert sha(cache / name) == value
    for name in ("best", "final"):
        assert sha(run / "global" / f"{name}.pt") == new[f"{name}_sha256"]
    assert new["reload_exact"]
    history = read(run / "global/history.json")
    assert [r["epoch"] for r in history] == list(range(1, 25))
    assert new["best_epoch"] == min(history, key=lambda r: r["validation"]["nll"])["epoch"]
    assert new["selected_consumed"] == new["best_epoch"] * 33696
    a, b = (
        read(reference / "matched-predictions.json"),
        read(run / "global/matched-predictions.json"),
    )
    for key, value in metrics(b).items():
        np.testing.assert_allclose(value, new["matched"][key], rtol=1e-5, atol=1e-7)
    comparisons = {"all": paired_image_metrics(a, b)}
    for kind in ("color", "texture", "count", "shape"):
        comparisons[kind] = paired_image_metrics(
            [r for r in a if r["taxonomy"] == kind], [r for r in b if r["taxonomy"] == kind]
        )
    payload = dict(
        config=config,
        result=result,
        reference=old,
        history=history,
        reference_history=read(reference / "history.json"),
        comparisons=comparisons,
        extraction=read(cache / "result.json"),
        encoder_environment=read(root / "full-image-environment-01.json"),
        environment_sha256=sha(root / "full-image-environment-01.json"),
        smoke=read(root / "shared-full-image-global-smoke-01/result.json"),
        hashes={str(p.relative_to(run)): sha(p) for p in run.rglob("*.json")},
    )
    out = Path("docs/pretrained_reuse/R2_FULL_IMAGE_RESULTS_20261012")
    write(out.with_suffix(".json"), payload)
    lines = [
        "# Full-image versus cropped global readout",
        "",
        "Fixed seed17,24epochs completed. Same initialization and full question order",
        "as the cropped reference, with exact best reload. All hashes checked.",
        f"Run source `{config['source_commit']}`; exact argv/config in companion JSON.",
        "Only preprocessing policy changed: full image bilinear288x288 versus",
        "resize-short-side288 then center crop. Wider field of view, aspect ratio",
        "and effective horizontal resolution change together; no isolated crop-cause claim.",
        "",
        "8,424 training images/33,696 questions;512 reused selector images/2,048 questions.",
        "808,704 encounters,775,008 repetitions. No reporting images or sealed audit.",
        "1,720,576 trainable parameters,133,475,584 total image-stack parameters.",
        "",
        "| Policy | Best epoch | Selected encounters | Acc | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("crop", old), ("full", new)):
        m = row["matched"]
        lines.append(
            f"| {name} | {row['best_epoch']} | {row['selected_consumed']} | "
            f"{m['accuracy']:.6f} | {m['nll']:.6f} | {m['brier']:.6f} | "
            f"{m['ece_15_bins']:.6f} |"
        )
    lines += [
        "",
        "## Task metrics",
        "",
        "| Policy | Task | Acc | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, row in (("crop", old), ("full", new)):
        for kind, m in row["matched_taxonomies"].items():
            lines.append(
                f"| {name} | {kind} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += [
        "",
        "## Paired image differences (full minus crop)",
        "",
        "512-image groups,10,000 resamples,seed17,linear percentile95% intervals.",
        "Reused validation and checkpoint selection limit inference; not blind testing.",
        "Intervals are descriptive, without correction for multiple task comparisons.",
        "",
        "| Scope | Metric | Difference | Lower | Upper |",
        "|---|---|---:|---:|---:|",
    ]
    for kind, comparison in comparisons.items():
        for metric, m in comparison["metrics"].items():
            lo, hi = m["percentile_95_interval"]
            lines.append(f"| {kind} | {metric} | {m['difference']:.6f} | {lo:.6f} | {hi:.6f} |")
    lines += [
        "",
        "## Full-image learning curve",
        "",
        "| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in history:
        t, v = r["train"], r["validation"]
        lines.append(
            f"| {r['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
            f"{t['nll']:.6f} | {v['nll']:.6f} |"
        )
    rotation = new["rotated"]
    lines += [
        "",
        "## Resources and controls",
        "",
        f"Extraction{payload['extraction']['seconds']:.3f}s CPU; readout"
        f"{result['total_seconds']:.3f}s; peak allocated VRAM{result['peak_vram_bytes']:,}bytes.",
        f"Fixed rotated-image Accuracy{rotation['accuracy']:.6f},NLL{rotation['nll']:.6f}.",
        "All gradients finite; best and final stored separately. Local compute only,$0 cloud.",
        f"Artifacts: `{run}` and `{cache}`. Existing artifacts unchanged.",
        "No arbitrary task95% or product-quality claim; other modality weaknesses remain.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
