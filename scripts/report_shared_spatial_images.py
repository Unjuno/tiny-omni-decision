"""Verify and summarize shared spatial image runs on existing selector IDs only."""

import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import paired, read
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


def metrics(rows):
    probs = [r["probabilities"] for r in rows]
    targets = [r["options"].index(r["target"]) for r in rows]
    for row, p in zip(rows, probs, strict=True):
        if row["prediction"] != row["options"][int(np.argmax(p))]:
            raise ValueError("stored prediction differs from probabilities")
        if not np.isfinite(p).all() or min(p) < 0 or abs(sum(p) - 1) > 1e-6:
            raise ValueError("invalid probabilities")
    return dict(count=len(rows),
                accuracy=sum(r["target"] == r["prediction"] for r in rows) / len(rows),
                nll=negative_log_likelihood(probs, targets), brier=brier_score(probs, targets),
                ece_15_bins=expected_calibration_error(probs, targets),
                mean_confidence=float(np.mean([max(p) for p in probs])))


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "shared-spatial-image-readout-01"
    cache = root / "videoprism-spatial-image-cache-01"
    config, result = read(run / "config.json"), read(run / "result.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    assert config["epochs"] == 24 and config["seed"] == 17
    for name, expected in config["cache_hashes"].items():
        assert sha(cache / name) == expected
    assert sha(run / "option-embeddings.npy") == config["option_embeddings_sha256"]
    histories, predictions = {}, {}
    for condition, r in result["conditions"].items():
        out = run / condition
        history = read(out / "history.json")
        assert [x["epoch"] for x in history] == list(range(1, 25))
        assert r["reload_exact"] and r["consumed"] == 808704
        assert r["selected_consumed"] == r["best_epoch"] * 33696
        assert r["best_epoch"] == min(history, key=lambda x: x["validation"]["nll"])["epoch"]
        for name in ("best", "final"):
            assert sha(out / f"{name}.pt") == r[f"{name}_sha256"]
        predictions[condition] = read(out / "matched-predictions.json")
        computed = metrics(predictions[condition])
        for key, value in computed.items():
            np.testing.assert_allclose(value, r["matched"][key], rtol=1e-5, atol=1e-7)
        histories[condition] = history
    a, b = result["conditions"].values()
    assert a["initial_tensor_hash"] == b["initial_tensor_hash"]
    assert a["order_sha256"] == b["order_sha256"]
    reference_paths = {
        "dino_spatial": root / "dinov2-spatial-bilinear-01/spatial/matched-predictions.json",
        "videoprism_pooled": root / (
            "image-videoprism-clevr4-full-seed17/checkpoint-selection-predictions.jsonl"),
    }
    refs = {name: (read(path) if path.suffix == ".json" else
                   [json.loads(line) for line in path.read_text().splitlines()])
            for name, path in reference_paths.items()}
    comparisons = {"spatial_minus_global": paired(predictions["global"], predictions["spatial"])}
    comparisons.update({f"{condition}_minus_{name}": paired(ref, pred)
                        for condition, pred in predictions.items() for name, ref in refs.items()})
    out = Path("docs/pretrained_reuse/R2_SHARED_SPATIAL_RESULTS_20261012")
    payload = dict(config=config, result=result, extraction=read(cache / "result.json"),
                   histories=histories, comparisons=comparisons,
                   references={name: dict(metrics=metrics(rows),
                       taxonomies={t: metrics([r for r in rows if r["taxonomy"] == t])
                                   for t in sorted({r["taxonomy"] for r in rows})},
                       sha256=sha(reference_paths[name])) for name, rows in refs.items()},
                   smoke=read(root / "shared-spatial-image-smoke-01/result.json"),
                   run_hashes={str(p.relative_to(run)): sha(p) for p in run.rglob("*.json")})
    write(out.with_suffix(".json"), payload)
    lines = ["# Shared VideoPrism spatial image comparison", "",
             "Two fresh seed17 runs completed24epochs. Encoder and text weights frozen;",
             "same initialization/question order verified, best reload exact, best/final",
             "checkpoint and cache hashes verified. No reporting or sealed audit records.",
             f"Source commit `{config['source_commit']}`; exact argv and hashes in companion JSON.",
             "",
             "Train8,424 images/33,696 questions; validation512 images/2,048 questions.",
             "Each run808,704 encounters,775,008 repeats; these are not new examples.",
             "Earliest minimum validation NLL selects best; final weights retained separately.",
             "1,720,576 trained parameters;133,475,584 total image stack including full",
             "VideoPrism114,365,184 and MiniLM17,389,824. This is not a complete Omni runtime.",
             "",
             "| Condition | Best epoch | Selected encounters | Accuracy | NLL | Brier | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in result["conditions"].items():
        m = r["matched"]
        lines.append(f"| {name} | {r['best_epoch']} | {r['selected_consumed']} | "
                     f"{m['accuracy']:.6f} | {m['nll']:.6f} | {m['brier']:.6f} | "
                     f"{m['ece_15_bins']:.6f} |")
    for name, r in payload["references"].items():
        m = r["metrics"]
        lines.append(f"| reference: {name} | — | — | {m['accuracy']:.6f} | "
                     f"{m['nll']:.6f} | {m['brier']:.6f} | {m['ece_15_bins']:.6f} |")
    lines += ["", "## Task metrics", "",
              "| Condition | Task | Accuracy | NLL | Brier | ECE |",
              "|---|---|---:|---:|---:|---:|"]
    for name, r in result["conditions"].items():
        for task, m in r["matched_taxonomies"].items():
            lines.append(f"| {name} | {task} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                         f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |")
    lines += ["", "## Complete fixed-budget curves", "",
              "| Condition | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, history in histories.items():
        for h in history:
            t, v = h["train"], h["validation"]
            lines.append(f"| {name} | {h['epoch']} | {t['accuracy']:.6f} | "
                         f"{v['accuracy']:.6f} | {t['nll']:.6f} | {v['nll']:.6f} |")
    lines += ["", "## Paired-image accuracy differences", ""]
    for name, c in comparisons.items():
        lo, hi = c["percentile_95_interval"]
        lines.append(f"- {name}: {100*c['difference']:+.3f}pp,95% interval"
                     f"[{100*lo:+.3f},{100*hi:+.3f}]pp.")
    lines += ["", "Paired512-image bootstrap,10,000 draws,seed17,linear percentile quantiles.",
              "These intervals do not correct reused-validation/model-selection bias.",
              "The DINO and old pooled VideoPrism references differ in preprocessing,",
              "readout and/or encoder, so these are not single-factor backbone comparisons.",
              "For the inspected640x480 training image, resize to384x288 then crop288x288",
              "removes the outer25% of the original width. DINO used the full image resized",
              "to224x224. This is a possible information-loss confound, not a measured cause",
              "of count errors; no crop adjustment or further run is made in this comparison.",
              "The new one-frame intermediate representation does not claim equality to",
              "the legacy repeated-frame cache; the previous numerical parity gate failed.",
              "", "## Compute and image dependence", "",
              f"CPU extraction{payload['extraction']['seconds']:.3f}s; GPU readout wall time"
              f"{result['total_seconds']:.3f}s. Peak allocated GPU memory"
              f"{result['peak_vram_bytes']:,}bytes; {result['gpu']}; cloud cost$0."]
    for name, r in result["conditions"].items():
        m = r["rotated"]
        lines.append(f"- {name}, fixed one-image rotation: Accuracy{m['accuracy']:.6f},"
                     f"NLL{m['nll']:.6f},Brier{m['brier']:.6f},ECE{m['ece_15_bins']:.6f}.")
    lines += ["", f"Artifacts: `{run}` and `{cache}`. Old artifacts remain unchanged.",
              "All best/final weights remain outside Git. This narrow attribute/counting",
              "comparison does not establish general image reasoning, no-text operation,",
              "simultaneous Omni fusion, or95–98% quality. No product promotion."]
    lines += ["", "## Interpretation and next decision", "",
              "Spatial retention adds only3 net correct answers out of2,048: its0.146pp",
              "aggregate difference is uncertain. Count Accuracy improves66.41%->68.75%",
              "but texture falls97.27%->95.51%. Brier improves while NLL/ECE worsen.",
              "Do not claim that spatial retention is universally better or adopt it.",
              "",
              "The spatial best has train95.50% versus validation90.92% Accuracy and",
              "NLL0.120454 versus0.227901. Count train82.48% versus validation68.75%",
              "shows a substantial generalization gap. Global has a smaller gap:",
              "train91.43% versus validation90.77%, NLL0.198187 versus0.224677.",
              "Spatial final24 is worse than best22; global selects24 and is not proven",
              "plateaued. The fixed budget ends here; no additional epochs or seeds.",
              "",
              "Both shared representations beat the legacy pooled VP result on these",
              "reused IDs, with positive paired accuracy intervals. DINO differences",
              "remain uncertain, and DINO spatial counting70.51% exceeds both new arms.",
              "This supports further shared-tower integration checks, not general-image",
              "superiority. The remaining counting weakness cannot be assigned solely",
              "to crop, representation, capacity, or training budget by this experiment.",
              "",
              "Preserve both alternatives and existing DINO. Prefer the global arm as",
              "the conservative integration reference because its NLL/ECE and gap are",
              "better, while explicitly retaining the count trade-off. A combined",
              "Omni runtime and its multi-input quality remain unverified. No95% claim.",
              "",
              "Verification:569 CPU tests pass,1 skipped; Ruff and all workflow",
              "manifest/catalog validation commands pass locally. Final push CI is",
              "checked independently. No paid compute or legacy artifact overwrite."]
    out.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
