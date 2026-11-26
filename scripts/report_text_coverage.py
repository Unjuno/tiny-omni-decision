"""Report completed coverage continuation without additional model evaluation."""

from pathlib import Path


def verify_finished(result, history):
    if (
        result.get("status") != "PASS_EXECUTION_ONLY"
        or result.get("smoke") is not False
        or result.get("steps") != 2048
        or result.get("reload_exact") is not True
        or [r["step"] for r in history] != list(range(0, 2049, 128))
    ):
        raise ValueError("run is not verified complete")
    if result["best_step"] != min(history, key=lambda r: r["selector"])["step"]:
        raise ValueError("wrong selection")


def main():
    from scripts.compare_numeric_embedding_predictions import compare
    from scripts.range_generalization import sha, write
    from scripts.report_numeric_deterministic import compact_config
    from scripts.report_spatial_curve import read

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/text-coverage-training-01")
    result = read(root / "result.json")
    history = read(root / "history.json")
    config = read(root / "config.json")
    verify_finished(result, history)
    for suffix in ["best", "final"]:
        if sha(root / f"{suffix}-delta.pt") != result[f"{suffix}_sha256"]:
            raise ValueError("weight hash mismatch")
    if len(config["train_ids"]) != 16384 or len(set(config["train_ids"])) != 16384:
        raise ValueError("training accounting mismatch")
    parent = root.parent / "numeric-deterministic-control-01"
    reference = next(r for r in read(parent / "history.json") if r["step"] == 1664)
    if config["validation_ids"] != read(parent / "config.json")["validation_ids"]:
        raise ValueError("validation IDs differ")
    paired = {}
    for source in ["original", "curriculum"]:
        initial = read(root / f"predictions-0-{source}.json")
        for metric in ["accuracy", "nll", "brier", "ece_15_bins"]:
            if abs(history[0][source][metric] - reference[source][metric]) > 1e-6:
                raise ValueError("initialization metric mismatch")
        for label, step in [("selected", result["best_step"]), ("final", 2048)]:
            paired[label + "-" + source] = compare(
                initial, read(root / f"predictions-{step}-{source}.json")
            )
    selected = next(r for r in history if r["step"] == result["best_step"])
    out = Path("docs/pretrained_reuse/R2_TEXT_COVERAGE_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=compact_config(config),
            history=history,
            paired=paired,
            hashes={p.name: sha(p) for p in root.iterdir() if p.is_file()},
        ),
    )
    lines = [
        "# Broader original-text coverage result",
        "",
        "Execution completed; development validation only.",
        "Optimizer reset from control step1664; no exact resume. All old outputs preserved.",
        "No isolated coverage-effect claim: training exposure and total updates both changed.",
        "",
        "| Step | Selector | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in history:
        a, b = r["original"], r["curriculum"]
        lines.append(
            f"| {r['step']} | {r['selector']:.6f} | {a['accuracy']:.6f} | {a['nll']:.6f} "
            f"| {b['accuracy']:.6f} | {b['nll']:.6f} |"
        )
    lines += [
        "",
        "## Selected checkpoint",
        "",
        "| Source/type | Accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in {
        **{s: selected[s] for s in ["original", "curriculum"]},
        **selected["tasks"],
    }.items():
        lines.append(
            f"| {name} | {m['accuracy']:.6f} | {m['nll']:.6f} "
            f"| {m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
        )
    lines += [
        "",
        f"Best step{result['best_step']}; selected encounters{selected['consumed']}"
        f", unique states{selected['unique_states']}.",
        "Full run16384 unique questions (8192 original+8192 curriculum); no within-run repeats.",
        "Original rows were seen in inherited final-block training; curriculum was reused.",
        f"Total loop{result['seconds']:.3f}s"
        f", evaluation{sum(r['evaluation_seconds'] for r in history):.3f}s,",
        f"peak allocatedVRAM{result['peak_vram_bytes']}bytes."
        " Cloud charges0; electricity unmeasured.",
        "Total22713601 parameters; trainable10647169. Best/final separate; selected reload exact.",
        "All probability rows and paired state-bootstrap comparisons are in JSON/local artifacts.",
        "Bootstrap10000 draws/seed17/percentile95%; selection validation is not blind.",
        "Rolling training CE uses changing examples; it is not a fixed-sample generalization gap.",
        "No transfer/audit evaluation or automatic adoption. Other modalities not re-evaluated.",
        f"Artifact `{root.as_posix()}`; source `{config['source_commit']}`.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
