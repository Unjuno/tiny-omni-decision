"""Summarize completed fixed decoder experiments without loading models or media."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    names = [
        "small-decoder-learning-01",
        "small-decoder-train-fit-01",
        "decoder-transfer-eval-01",
    ]
    data = {}
    for name in names:
        folder = root / name
        result = json.loads((folder / "result.json").read_text())
        if result["status"] not in {"PASS", "PASS_EXECUTION_ONLY"}:
            raise ValueError(name)
        data[name] = {
            "result": result,
            "hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(folder.glob("*.json"))
            },
        }
    run = data[names[0]]
    for filename in ["config", "history"]:
        run[filename] = json.loads((root / names[0] / f"{filename}.json").read_text())
    # Full ordered IDs remain in the hashed local config, not the Git summary.
    for field in ["train_ids", "validation_ids"]:
        ids = run["config"].pop(field)
        run["config"][field + "_summary"] = {
            "counts": [len(group) for group in ids] if field == "validation_ids" else len(ids),
            "ordered_json_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    history = run["history"]
    selected = min(history, key=lambda row: row["selector"])
    assert selected["step"] == run["result"]["best_step"] == 2048
    assert run["result"]["reload_exact"]
    out = Path("docs/pretrained_reuse/R2_SMALL_DECODER_RESULTS_20261012")
    out.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Small decoder learning and fixed transfer results",
        "",
        "Execution PASS; quality objective unmet. No product promotion or sealed audit.",
        "Fresh pinned SmolLM2-135M-Instruct; 134,515,008 resident parameters before",
        "quantization. Only last four decoder blocks and final norm trained:",
        "14,160,960 parameters. FP32/SDPA, seed17, AdamW constant2e-5, batch8,",
        "2,048 updates. Full prompts, no truncation. See the frozen learning,",
        "training-fit and transfer protocols for exact inputs and selection rules.",
        "",
        f"Source commit: `{run['config']['source_commit']}`.",
        f"Start: {run['config']['started_utc']}; end: {run['result']['finished_utc']}.",
        "Normal scheduled completion. Best is earliest minimum mean validation NLL",
        "across original and curriculum; best and final both step2,048, separate files.",
        "Their serialization hashes differ; best reload predictions agree exactly.",
        "Frozen tensor hash matches before/after. Weights-only, not exact resume.",
        "",
        "## Learning curve",
        "",
        "| Step | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Selector |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        a, b = row["original"], row["curriculum"]
        lines.append(
            f"| {row['step']} | {a['accuracy']:.6f} | {a['nll']:.6f} | "
            f"{b['accuracy']:.6f} | {b['nll']:.6f} | {row['selector']:.6f} |"
        )
    lines += [
        "",
        "Whole run and selected checkpoint each consumed16,384 unique questions,",
        "8,192 per source,6,056 underlying states; zero repeated question IDs.",
        "Original validation2,540 and curriculum2,048 questions. Per-task metrics,",
        "Brier/ECE and all scheduled predictions are preserved in local artifacts.",
        "Original accuracy improves30.91%→49.92%; curriculum remains24.22%.",
        "Best selector is at the endpoint: no demonstrated global plateau.",
        "Curriculum fluctuates near chance, not evidence that general reasoning was learned.",
        "Rolling training CE is not a fixed-example generalization-gap estimate.",
        "",
        "## Fixed seen-training diagnostic",
        "",
        "First1,024 consumed questions,512/source, selected weights only. Original",
        "accuracy51.17%, curriculum22.27%. Low curriculum fit also occurs on seen",
        "examples. This supports a fitting/optimization/representation bottleneck",
        "within this run, not a unique capacity diagnosis or impossibility claim.",
        "Training curriculum ranges/wording differ from narrow validation, so their",
        "difference is not an identically distributed generalization gap.",
        "",
        "## Fixed new-state transfer",
        "",
        "128 new states,1,024 base questions, six paired arms;6,144 rows and5,632",
        "distinct contents. Known templates; development diagnostics, not blind tests.",
        "BERT has different pretraining and inherited training; not equal compute.",
        "",
        "| Model | Arm | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    transfer = data[names[2]]["result"]
    for model, variants in transfer["metrics"].items():
        for variant, item in variants.items():
            m = item["overall"]
            lines.append(
                f"| {model} | {variant} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += [
        "",
        "JSON includes every task, opposite-condition pair success, candidate/state",
        "order identity consistency and paired128-state bootstrap versus base",
        "(10,000 draws, seed17, percentile95%, no simultaneous correction).",
        "No checkpoints or temperatures selected from transfer outcomes.",
        "Adapted decoder base-arm accuracy23.44% versus pretrained22.07%:",
        "difference+1.37pp, paired95% interval[-2.25,+4.88]pp. Every decoder",
        "arm's interval versus pretrained includes zero. The BERT reference gets",
        "81.05% on the base arm but38.18% on domain terms and31.84% when expanded.",
        "Candidate reversal preserves selected identity only12.60% for adapted",
        "decoder, versus100% for independent-candidate BERT. This is a major",
        "ordering weakness in this decoder pipeline, not general decision quality.",
        "Keep BERT as the stronger measured text reference, without claiming95%.",
        "",
        "## Resources and preservation",
        "",
    ]
    for name in names:
        r = data[name]["result"]
        lines.append(
            f"- `{name}`: {r['seconds']:.3f}s, peak allocated VRAM"
            f" {r['peak_vram_bytes']:,} bytes."
        )
    evaluation_seconds = sum(r["evaluation_seconds"] for r in history)
    lines += [
        f"- Scheduled validation total: {evaluation_seconds:.3f}s; exact reload adds",
        "  another full validation within total run time. Remaining time includes",
        "  training, saving and hashing, not a pure training-only measurement.",
        "- RTX3080Laptop16GB; local compute, cloud cost$0. Environment pinned by",
        "  feasibility record; no package upgrade during this run.",
        "- Prior checkpoints/corpora preserved. No sensor, image/audio/video training",
        "  in this experiment; no quantization, no final sealed evaluation.",
        "- All files live under `C:/CodexArtifacts/pretrained-reuse-r2/` plus the",
        "  three run names above. JSON records config/history/prediction hashes;",
        "  result records best/final tensor-file hashes. Large weights stay outside Git.",
        "- Ruff passes;374 CPU tests pass,1 skipped; model/dataset/catalog",
        "  manifest validation passes. GitHub status must be checked on the pushed commit.",
        "",
        "Four-modality95–98% remains unmet. Do not replace the smaller BERT reference",
        "based on parameter budget alone. Further architecture changes require a",
        "separate predeclared diagnostic; this run does not justify an automatic",
        "larger budget or repeated evaluation of the same transfer probe.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
