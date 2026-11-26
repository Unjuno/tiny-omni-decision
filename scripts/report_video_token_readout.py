"""Summarize completed token readout runs without further inference."""

import json
from pathlib import Path

from scripts.range_generalization import sha, write


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/video-token-readout-01")
    result, config = read(root / "result.json"), read(root / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    histories = {name: read(root / name / "history.json") for name in result["conditions"]}
    assert all(len(h) == 24 for h in histories.values())
    extraction = read(root / "extraction.json")
    assert len(extraction) == 456 and len({r["media_sha256"] for r in extraction}) == 456
    assert max(r["pooled_max_abs"] for r in extraction) <= 1e-5
    # Labels in the two prediction files describe the condition; compare numeric outputs.
    a, b = (read(root / "no_video" / p) for p in ["predictions.json", "predictions_shuffled.json"])
    for x, y in zip(a, b, strict=True):
        assert x["id"] == y["id"] and x["probabilities"] == y["probabilities"]
    config["id_counts"] = {s: len(ids) for s, ids in config.pop("ids").items()}
    out = Path("docs/pretrained_reuse/R2_TOKEN_READOUT_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=config,
            histories=histories,
            hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.json"))},
        ),
    )
    lines = [
        "# Frozen video token readout comparison",
        "",
        "Execution completed; previously observed development validation, not blind quality.",
        f"Source `{config['source_commit']}`. Seed17; 24epochs each; fixed split and order.",
        "3794 unique training choices/256 scenes; 3014 validation choices/200 scenes.",
        "Each condition sees 91,056 encounters (87,262 repeats). No sealed audit access.",
        "456 full FP32 token tensors extracted transiently; every pooled mean matches prior cache.",
        "",
        "| Mode | Type | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode, r in result["conditions"].items():
        for kind, m in {
            "macro": r["matched"]["macro_question_type"],
            **r["matched"]["by_question_type"],
        }.items():
            lines.append(
                f"| {mode} | {kind} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += ["", "## Paired scene uncertainty", ""]
    for name, value in result["comparisons"].items():
        low, high = value["percentile_95_interval"]
        lines.append(
            f"- {name}: {value['difference'] * 100:+.3f}pp; "
            f"95% interval [{low * 100:+.3f}, {high * 100:+.3f}]pp."
        )
    lines += [
        "",
        "200-scene paired percentile bootstrap, 10,000 draws, seed17;",
        "intervals do not remove validation selection bias or multiple comparisons.",
        "",
        "## Learning curves",
        "",
        "| Mode | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, history in histories.items():
        for row in history:
            t, v = row["train"]["macro_question_type"], row["validation"]["macro_question_type"]
            lines.append(
                f"| {mode} | {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
                f"{t['nll']:.6f} | {v['nll']:.6f} |"
            )
    lines += ["", "## Resources and artifacts", ""]
    for mode, r in result["conditions"].items():
        lines.append(
            f"- {mode}: best epoch {r['best_epoch']}, selected encounters "
            f"{r['selected_consumed']}, total parameters {r['parameters']}, "
            f"active parameters {r['active_parameters']}, {r['seconds']:.3f}s."
        )
    lines += [
        f"- Extraction {result['extraction_seconds']:.3f}s; "
        f"total {result['total_seconds']:.3f}s; peak allocated VRAM "
        f"{result['peak_vram_bytes']} bytes; local cost $0.",
        "- Frozen encoders: VideoPrism 114,365,184 + MiniLM 17,389,824 parameters.",
        "  Mean/zero-video controls carry unused key/position parameters; active capacity differs.",
        "- Best/final weights separate; best reload exact. Original artifacts preserved.",
        f"- Artifacts `{root.as_posix()}`; full probabilities, histories and hashes retained.",
        "- smoke-01 stopped before training: one validation scene cannot be deranged.",
        "  smoke-02 used first/last rows across task types, completed all three conditions.",
        "- Four-modality95% and general decision quality remain unproven.",
    ]
    lines += [
        "",
        "## Decision and interpretation",
        "",
        "Do not adopt this attention readout: Accuracy is 0.282pp below its matched mean",
        "control, and the paired interval spans zero. Its lower NLL/Brier/ECE versus mean",
        "does not establish a reliable gain over no-video; no-video NLL is lower still.",
        "Scene derangement does not hurt Accuracy reliably (matched-minus-shuffled -0.964pp).",
        "This does not prove video information is absent in the pretrained tokens.",
        "",
        "At selected attention epoch21, train Accuracy57.57% versus validation52.44%",
        "and train NLL0.67876 versus validation0.69273 indicate limited train fit plus a gap.",
        "At epoch24 both train and validation deteriorate; do not label this final change",
        "pure overfitting while ignoring optimization instability. Best21 is retained.",
        "The single attention readout does not explicitly model object-to-object relations.",
        "These data do not isolate optimization, task/label complexity, text conditioning,",
        "or representational limitations. No backbone capacity ceiling has been established.",
        "",
        "Full video/text/attention parameter count is132,148,992 before quantization.",
        "This excludes any audio encoder; it is not a complete omni-model size claim.",
        "The logged active_parameters field counts parameters in the chosen forward branch.",
        "For no-video, value.weight (98,304 parameters) receives zero input and has no",
        "functional influence, despite being included in that field and optimizer group.",
        "Thus functionally influential no-video parameters are196,864. No effective-capacity",
        "equivalence claim is made. Mean and attention share identical query/value initialization",
        "by resetting CPU RNG before construction; the extra keys and positions are explicit.",
        "",
        "No automatic epoch extension. Before another full validation search, diagnose whether",
        "the readout can fit a fixed training-only subset and whether its gradients/attention",
        "use video. Such a diagnostic must retain original data and use a new output path.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
