"""Publish every fixed transfer arm without new model inference."""

from pathlib import Path

from scripts.build_vocabulary_transfer import VARIANTS
from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/coverage-transfer-eval-01")
    result = read(root / "result.json")
    completion = read(root / "completion.json")
    protocol = read(root / "protocol.json")
    pairs = read(root / "opposing-pairs.json")
    selection = read(root.parent / "coverage-transfer-eval-01-selection.json")
    if result["status"] != "PASS_EXECUTION_ONLY" or completion != {
        "status": "PASS",
        "frozen_sources_verified": True,
        "opposing_pairs_verified": True,
    }:
        raise ValueError("incomplete evaluation")
    if protocol["variants"] != list(VARIANTS) or protocol["expected_rows"] != 10240:
        raise ValueError("altered arms")
    for name in ["parent", "candidate"]:
        if set(result["metrics"][name]) != set(VARIANTS):
            raise ValueError("missing arm")
        for variant in VARIANTS:
            if result["metrics"][name][variant]["overall"]["count"] != 1024:
                raise ValueError("missing rows")
            if len(read(root / f"{name}-{variant}-predictions.json")) != 1024:
                raise ValueError("missing predictions")
    out = Path("docs/pretrained_reuse/R2_COVERAGE_TRANSFER_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            completion=completion,
            protocol=protocol,
            selection=selection,
            opposing_pairs=pairs,
            hashes={p.name: sha(p) for p in root.iterdir() if p.is_file()},
        ),
    )
    lines = [
        "# Coverage continuation: fixed transfer results",
        "",
        "Execution completed; inference only. No checkpoint selection from these results.",
        "Same128 latent states/1024 base questions,10arms/10240rows per model.",
        "Repeated variants do not multiply independent sample size. Observed development data.",
        "Families0–3 use trained comparison words; family4 also changes domain field names.",
        "Expanded values use7*x+19. No dropped arms, prompt search, calibration or new training.",
        "",
        "| Arm | Old Acc | New Acc | Old NLL | New NLL | Accuracy difference95%CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        a = result["metrics"]["parent"][variant]["overall"]
        b = result["metrics"]["candidate"][variant]["overall"]
        lo, hi = result["comparisons"][variant]["percentile_95_interval"]
        lines.append(
            f"| {variant} | {a['accuracy']:.6f} | {b['accuracy']:.6f} | {a['nll']:.6f}"
            f" | {b['nll']:.6f} | [{lo:+.6f},{hi:+.6f}] |"
        )
    lines += [
        "",
        "| Arm | Old Brier | New Brier | Old ECE | New ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        a = result["metrics"]["parent"][variant]["overall"]
        b = result["metrics"]["candidate"][variant]["overall"]
        lines.append(
            f"| {variant} | {a['brier']:.6f} | {b['brier']:.6f} |"
            f" {a['ece_15_bins']:.6f} | {b['ece_15_bins']:.6f} |"
        )
    lines += [
        "",
        "## Opposite-condition pairs",
        "",
        "| Model | Arm | Field | Both correct | Prediction changed | Pairs |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, arms in pairs.items():
        for variant, fields in arms.items():
            for field, m in fields.items():
                lines.append(
                    f"| {name} | {variant} | {field} | {m['both_correct']} |"
                    f" {m['prediction_changed']} | {m['pairs']} |"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        "Training-validation improvement did not close domain-word or expanded-range fragility.",
        "Base-family4 stays near40%; expanded-family4 reaches36.13% but NLL worsens",
        "1.71050 to2.06123 and ECE0.26242 to0.32043. Higher Accuracy alone is insufficient.",
        "Expanded-family0–3 probability scores generally fail to improve; preserve all"
        " regressions.",
        "Do not adopt the checkpoint as a general reasoning fix or replace shared"
        " multimodal towers.",
        "This is not evidence that the model family can never improve. No clean capacity ceiling.",
        "The family4 change combines vocabulary/field naming; it cannot isolate either cause.",
        "The next quality experiment needs a different, preregistered hypothesis; no"
        " automatic extension.",
        "",
        "## Provenance and limits",
        "",
        f"Source `{selection['source_commit']}`; checkpoint hashes and input hash in"
        f" accompanying JSON.",
        "Both full training corpora were checked for ID, normalized-state and"
        " latent-state overlap:zero.",
        "Every symbolic target was verified; all frozen source hashes match after evaluation.",
        "Paired128-state bootstrap10000draws/seed20261012, percentile95%, no"
        " multiplicity correction.",
        "Repeated development evaluation and prior checkpoint selection prevent blind-test claims.",
        f"Timed evaluation{result['seconds']:.3f}s; peak"
        f" allocatedVRAM{result['peak_vram_bytes']}bytes.",
        "Cloud charges0; electricity unmeasured. Previous artifacts untouched.",
        f"Artifacts `{root.as_posix()}`. No sealed audit or other modality inference.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
