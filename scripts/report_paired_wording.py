"""Report the completed matched experiment; never load models or audit data."""

import json
from pathlib import Path

from scripts.compare_numeric_embedding_predictions import compare
from scripts.range_generalization import sha, write
from scripts.verify_wording_result import verify_pair


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    runs = [root / name for name in ("paired-wording-control-01", "paired-wording-treatment-01")]

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    triples = [
        tuple(read(p / f"{name}.json") for name in ("config", "result", "history")) for p in runs
    ]
    verify_pair(*triples)
    for run, (_, result, _) in zip(runs, triples, strict=True):
        for which in ("best", "final"):
            if sha(run / f"{which}-delta.pt") != result[f"{which}_sha256"]:
                raise ValueError("artifact hash mismatch")
    for step in (0, 128, 256):
        for source in ("original", "curriculum"):
            name = f"predictions-{step}-{source}.json"
            if (runs[0] / name).read_bytes() != (runs[1] / name).read_bytes():
                raise ValueError("pre-intervention predictions differ")
    comparisons = {}
    for label in ("selected", "endpoint"):
        steps = [t[1]["best_step"] if label == "selected" else 1024 for t in triples]
        for source in ("original", "curriculum"):
            comparisons[label + "-" + source] = compare(
                *[
                    read(run / f"predictions-{step}-{source}.json")
                    for run, step in zip(runs, steps, strict=True)
                ]
            )
    compact = []
    for run, (config, result, history) in zip(runs, triples, strict=True):
        compact.append(
            dict(
                path=str(run),
                config={
                    k: v
                    for k, v in config.items()
                    if k not in ("parent_ids", "train_ids", "validation_ids")
                },
                result=result,
                history=history,
                hashes={p.name: sha(p) for p in run.iterdir() if p.is_file()},
            )
        )
    path = Path("docs/pretrained_reuse/R2_PAIRED_WORDING_RESULTS_20261012")
    if path.with_suffix(".json").exists():
        raise FileExistsError("preserve existing report")
    write(
        path.with_suffix(".json"),
        dict(runs=compact, comparisons=comparisons, prefix_prediction_parity_exact=True),
    )
    lines = [
        "# Matched wording-view learning results",
        "",
        "Both 1,024-update runs completed with exact best reload and frozen-weight checks.",
        "Parent order and fixed conditions match; step0/128/256 predictions are byte-identical.",
        "Development validation only. No sealed audit, product promotion or general95% claim.",
        "",
        "| Arm | Step | Selector | Original Accuracy | Curriculum Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for config, result, history in triples:
        if config["arm"] == "wording":
            lines += [
                "| Arm | Step | Selector | Original Accuracy | Curriculum Accuracy |",
                "|---|---:|---:|---:|---:|",
            ]
        for h in history:
            lines.append(
                f"| {config['arm']} | {h['step']} | {h['selector']:.6f} | "
                f"{h['original']['accuracy']:.6f} | {h['curriculum']['accuracy']:.6f} |"
            )
        selected = next(h for h in history if h["step"] == result["best_step"])
        lines += [
            "",
            f"{config['arm']}: best step{result['best_step']}; "
            f"selected encounters{selected['consumed']}, "
            f"unique parents{selected['unique_parent_questions']}. "
            "Full run8192 encounters/5120 unique parents;3072 repeated parent encounters.",
            f"Timed {result['seconds']:.3f}s; peak allocated VRAM "
            f"{result['peak_vram_bytes']}bytes.",
            "",
        ]
    lines += [
        "## Selected and endpoint metrics",
        "",
        "| Arm | Checkpoint | Source | Accuracy | NLL | Brier | ECE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for config, result, history in triples:
        for label, step in (("selected", result["best_step"]), ("endpoint", 1024)):
            h = next(h for h in history if h["step"] == step)
            for source in ("original", "curriculum"):
                m = h[source]
                lines.append(
                    f"| {config['arm']} | {label} ({step}) | {source} | "
                    f"{m['accuracy']:.6f} | {m['nll']:.6f} | "
                    f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
                )
    lines += [
        "",
        "Paired state bootstrap:10,000 draws, seed17, percentile95% intervals in JSON.",
        "Selected checkpoint metrics reuse selection validation; they are not blind estimates.",
        "Rolling training CE averages changing examples and is not a fixed train/validation gap.",
        "Cloud cost0; local electricity unmeasured. Full source/config/artifact hashes in JSON.",
        "No new image/audio/video measurement. Wording augmentation changes syntax and synonyms",
        "jointly; no individual word-level causal claim. Existing artifacts remain preserved.",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
