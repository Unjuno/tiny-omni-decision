"""Compare the numeric-embedding run with the identical parent2048-step prefix."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "numeric-embedding-long-01"
    parent = root / "conditional-repair-comparison-vocabulary-01"
    result = read(run / "result.json")
    config = read(run / "config.json")
    history = read(run / "history.json")
    old = [r for r in read(parent / "history.json") if r["step"] <= 2048]
    assert result["status"] == "PASS" and result["steps"] == 2048 and not result["smoke"]
    assert [r["step"] for r in history] == list(range(0, 2049, 128))
    assert config["train_ids"] == read(parent / "config.json")["train_ids"][:16384]
    assert config["validation_ids"] == read(parent / "config.json")["validation_ids"]
    for source in ["original", "curriculum"]:
        for metric in ["accuracy", "nll", "brier", "ece_15_bins"]:
            assert abs(history[0][source][metric] - old[0][source][metric]) <= 1e-6
    best = min(history, key=lambda r: r["selector"])
    oldbest = min(old, key=lambda r: r["selector"])
    assert result["best_step"] == best["step"]
    out = Path("docs/pretrained_reuse/R2_NUMERIC_EMBEDDING_LONG_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            config=config,
            history=history,
            parent_prefix=old,
            parent_best=oldbest["step"],
            hashes={p.name: sha(p) for p in run.iterdir() if p.is_file()},
        ),
    )
    lines = [
        "# Matched2048-step numeric embedding result",
        "",
        "Execution PASS; frozen hashes and exact best reload verified by runner.",
        "Same initialization,16384-question prefix and validation IDs as parent. Zero-step",
        "Acc/NLL/Brier/ECE match within1e-6. Parent comparison restricted to0..2048.",
        "Original encoder-coverage warm start has inherited training; no exact optimizer resume.",
        "Only additional trainable scope is343 numeric embedding deltas /131712 parameters.",
        "Total22,845,313; trainable10,778,881. No new audit/transfer evaluation.",
        "",
        "| Run | Step | Selector | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in [("parent", old), ("numeric_delta", history)]:
        for r in rows:
            a, b = r["original"], r["curriculum"]
            lines.append(
                f"| {name} | {r['step']} | {r['selector']:.6f} | "
                f"{a['accuracy']:.6f} | {a['nll']:.6f} | "
                f"{b['accuracy']:.6f} | {b['nll']:.6f} |"
            )
    lines += [
        "",
        "## Selected checkpoints",
        "",
        "| Run | Best step | Source | Acc | NLL | Brier | ECE |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, r in [("parent", oldbest), ("numeric_delta", best)]:
        for source in ["original", "curriculum"]:
            m = r[source]
            lines.append(
                f"| {name} | {r['step']} | {source} | {m['accuracy']:.6f} | "
                f"{m['nll']:.6f} | {m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += [
        "",
        f"Selected exposure{best['consumed']} questions; full run16384; "
        f"selected unique states{best['unique_states']}.",
        "Per-source/task counts and gradients in JSON. Rolling CE is not a fixed-set gap.",
        f"Timed run{result['seconds']:.3f}s, peak VRAM{result['peak_vram_bytes']:,}bytes; cloud$0.",
        "Time excludes initial load/tokenization. Separate best/final weights saved locally.",
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-embedding-long-01`.",
        "No product adoption or95% claim; one seed, reused development selector.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
