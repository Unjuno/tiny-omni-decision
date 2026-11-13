"""Record the fixed rotation diagnostic and preserve provenance without model inference."""

import json
from pathlib import Path

from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "candidate-rotation-eval-01"
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "PASS_EXECUTION_ONLY"
    prior = json.loads((root / "decoder-transfer-eval-01/result.json").read_text())
    config = json.loads((run / "config.json").read_text())
    out = Path("docs/pretrained_reuse/R2_CANDIDATE_ROTATION_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=config,
            hashes={p.name: sha(p) for p in sorted(run.glob("*.json"))},
            single_forward=prior["metrics"]["decoder"],
        ),
    )
    lines = [
        "# Candidate-position balancing results",
        "",
        "Inference completed on the fixed selected decoder; no training or selection.",
        f"Source `{config['source_commit']}`; pinned base/delta/input hashes in JSON.",
        "134,515,008 resident parameters, FP32/SDPA, RTX3080Laptop16GB.",
        "All1,024 questions/128 states in each of two paired arms; previously observed",
        "development data, no blind/general-quality claim. Same state/question, all",
        "cyclic option positions, temperature1, identity-aligned arithmetic mean.",
        "",
        "| Arm | Scorer | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, detail in result["metrics"].items():
        for scorer, metrics in [
            ("single", prior["metrics"]["decoder"][arm]["overall"]),
            ("rotation mean", detail["overall"]),
        ]:
            lines.append(
                f"| {arm} | {scorer} | {metrics['accuracy']:.6f} | {metrics['nll']:.6f} | "
                f"{metrics['brier']:.6f} | {metrics['ece_15_bins']:.6f} |"
            )
    lines += ["", "## Paired Accuracy difference", ""]
    for arm, comparison in result["comparisons"].items():
        low, high = comparison["percentile_95_interval"]
        lines.append(
            f"- {arm}: {comparison['difference'] * 100:+.3f}pp; "
            f"95% interval[{low * 100:+.3f},{high * 100:+.3f}]pp."
        )
    lines += [
        "",
        "Paired128-state bootstrap,10,000 draws,seed17,percentile interval; no",
        "multiple-comparison correction. Calibration and all task metrics in JSON.",
        "",
        "Reversed-order identity consistency: "
        f"{result['candidate_reversal_identity_consistency']:.6f}.",
        "Cyclic averaging is not mathematically invariant to every permutation.",
        "Rotation0 maximum probability difference from saved results: "
        f"{result['rotation0_max_probability_difference']:.9g}.",
        "",
        "## Resource cost",
        "",
        f"- Wall time {result['seconds']:.3f}s; "
        f"peak allocated VRAM {result['peak_vram_bytes']:,} bytes.",
        f"- Forward example counts: `{json.dumps(result['counts'], sort_keys=True)}`.",
        "- Cloud cost$0; zero optimization updates. Full probabilities and each",
        "  individual rotation remain under "
        "`C:/CodexArtifacts/pretrained-reuse-r2/candidate-rotation-eval-01`.",
        "- Old corpora and model files unchanged; no sealed audit access.",
        "",
        "## Decision",
        "",
        "Do not adopt. Base-arm Accuracy falls23.44%→21.29%, while reversed-order",
        "Accuracy changes24.32%→24.71%. The gain is not consistent across arms.",
        "Lower NLL alone does not establish conditional reasoning; near-uniform",
        "probabilities can have low ECE while decisions remain near chance.",
        "This tests one deterministic positional balancing rule, not every possible",
        "readout or a model-family capacity limit. Do not launch another training",
        "budget from this result alone. Preserve the stronger small BERT reference",
        "and the four-modality95–98% objective, which remains unmet.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
