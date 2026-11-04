"""Compare the fresh deterministic control and numeric-delta runs."""

import hashlib
import json
from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read


def compact_config(config):
    """Keep ID order provenance without duplicating the full local config in Git."""

    def summary(ids):
        payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return dict(count=len(ids), order_sha256=hashlib.sha256(payload).hexdigest())

    result = dict(config)
    result["train_ids"] = summary(config["train_ids"])
    result["validation_ids"] = [summary(ids) for ids in config["validation_ids"]]
    return result


def verify_pair(control, candidate):
    """Reject unplanned configuration differences before comparing outcomes."""
    if control.get("condition") != "control" or candidate.get("condition") != "numeric":
        raise ValueError("incorrect comparison conditions")
    for config in (control, candidate):
        if config.get("steps") != 2048 or config.get("smoke") is not False:
            raise ValueError("incomplete experiment configuration")
        if config.get("deterministic") is not True:
            raise ValueError("non-deterministic configuration")
    allowed = {"argv", "started_utc", "condition", "trace_sha256", "disk_output_allowance"}
    a = {k: v for k, v in control.items() if k not in allowed}
    b = {k: v for k, v in candidate.items() if k not in allowed}
    if a != b:
        differing = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        raise ValueError(f"unplanned configuration differences: {differing}")


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "numeric-deterministic-candidate-01"
    parent = root / "numeric-deterministic-control-01"
    result = read(run / "result.json")
    config = read(run / "config.json")
    history = read(run / "history.json")
    old = read(parent / "history.json")
    parent_result = read(parent / "result.json")
    assert parent_result["status"] == "PASS" and parent_result["steps"] == 2048
    verify_pair(read(parent / "config.json"), config)
    from scripts.compare_numeric_embedding_predictions import compare

    paired = {}
    pairs = [(str(step), step, step) for step in [512, 1024, 1536, 1920, 2048]]
    pairs.append(("selected", parent_result["best_step"], result["best_step"]))
    for label, a, b in pairs:
        for source in ["original", "curriculum"]:
            paired[label + "-" + source] = compare(
                read(parent / f"predictions-{a}-{source}.json"),
                read(run / f"predictions-{b}-{source}.json"),
            )
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
    out = Path("docs/pretrained_reuse/R2_NUMERIC_DETERMINISTIC_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            parent_result=parent_result,
            paired_comparisons=paired,
            config_summary=compact_config(config),
            history=history,
            control_history=old,
            control_best=oldbest["step"],
            hashes={p.name: sha(p) for p in run.iterdir() if p.is_file()},
            control_hashes={p.name: sha(p) for p in parent.iterdir() if p.is_file()},
        ),
    )
    lines = [
        "# Fresh deterministic 2048-step numeric embedding comparison",
        "",
        "Execution PASS; frozen hashes and exact best reload verified by runner.",
        "Same initialization, 16,384-question order and validation IDs as fresh control. Zero-step",
        "Acc/NLL/Brier/ECE match within 1e-6. Both conditions ran 2048 updates.",
        "Original encoder-coverage warm start has inherited training; no exact optimizer resume.",
        "Only additional trainable scope is 343 numeric embedding deltas / 131,712 parameters.",
        "Both total 22,845,313; trainable control 10,647,169 vs numeric 10,778,881.",
        "No new audit/transfer evaluation. Historical nondeterministic runs are separate.",
        "Full configs stay immutable locally; ID lists in Git use counts and order SHA-256.",
        "",
        "| Run | Step | Selector | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in [("control", old), ("numeric_delta", history)]:
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
    for name, r in [("control", oldbest), ("numeric_delta", best)]:
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
        "Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-deterministic-candidate-01`.",
        "No product adoption or 95% claim; one seed, reused development selector.",
        f"Control selected exposure {oldbest['consumed']} questions; "
        f"full 16384; selected unique states {oldbest['unique_states']}.",
        f"Control timed run {parent_result['seconds']:.3f}s, "
        f"peak VRAM {parent_result['peak_vram_bytes']:,} bytes.",
    ]
    lines += [
        "",
        "## Paired state comparisons",
        "",
        "| Point/source | Accuracy delta |95% interval | NLL delta |",
        "|---|---:|---|---:|",
    ]
    for label, c in paired.items():
        a = c["metrics"]["accuracy"]
        n = c["metrics"]["nll"]
        lo, hi = a["percentile_95_interval"]
        lines.append(
            f"| {label} | {a['difference']:.6f} | [{lo:.6f},{hi:.6f}] | {n['difference']:.6f} |"
        )
    lines += [
        "",
        "State-cluster percentile bootstrap10000 draws,seed17; sample-weighted",
        "ratios preserve unequal original group sizes. Reused development data,",
        "no multiplicity correction or claim of a blind product result.",
    ]
    lines += [
        "",
        "## Interpretation",
        "",
        "Both independently select step1664. Numeric deltas improve curriculum NLL",
        "0.449164 to0.433914 and Accuracy81.54% to82.37%, while original-text",
        "Accuracy61.54% to61.26%, NLL0.885243 to0.890595, Brier0.494998 to0.497011",
        "and ECE0.067484 to0.075976 deteriorate. The selected Accuracy-difference",
        "interval crosses zero for both sources. Do not promote on the mean selector alone.",
        "",
        "At fixed2048, curriculum Accuracy improves1.27pp (interval+0.39 to+2.15pp),",
        "but original NLL worsens0.008158. These are repeated development comparisons,",
        "not a new blind test or proof that numeric embeddings are the sole bottleneck.",
        "The candidate's mean validation NLL worsens0.662255 at1664 to0.696826 at2048;",
        "keeping the last checkpoint would discard the selected validation result.",
        "Rolling training CE is cumulative over changing examples, so no fixed-train",
        "generalization-gap claim follows. This is a validation-degradation warning.",
        "",
        "Each run consumes16384 distinct question IDs/6056 states; selected13312/5507.",
        "These are per-run counts: prior inherited training and reused source records",
        "must not be counted as newly collected lifetime data. Source split is8192+8192",
        "for the full run and6656+6656 at selection. No repeats within this run schedule.",
        "",
        "Keep both artifacts, no product adoption, no automatic extension/seed search.",
        "The already frozen shared-text image comparison uses the control checkpoint",
        "selected before these candidate outcomes. No architecture/data/selector changed.",
        "The general four-modality95% target remains unmet; narrow numeric improvement",
        "does not resolve text semantics, transfer, video reasoning or joint fusion.",
        "",
        "Total timed local work for this pair2371.929s, excluding initial loading/tokenizing.",
        "Concurrent bounded CPU checks also ran; timings do not isolate hardware speed gains.",
        "No paid compute. Energy/electricity cost was not measured. Both best/final hashes",
        "and runtime settings are retained; weights-only artifacts are not strict resumes.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
