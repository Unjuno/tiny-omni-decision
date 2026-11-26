"""Summarize the completed alias/wording factorial without additional inference."""

import hashlib
import json
from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read
from scripts.transfer_factorial import VARIANTS, make_cells


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "transfer-factorial-eval-01"
    data = root / "transfer-factorial-inputs-01"
    r = read(run / "result.json")
    f = read(run / "factorial.json")
    p = read(run / "protocol.json")
    if r["status"] != "PASS_EXECUTION_ONLY" or f["status"] != "PASS":
        raise ValueError("not complete")
    if any(v != 0 for modes in f["parity_max_abs"].values() for v in modes.values()):
        raise ValueError("expected measured exact replay")
    old = [
        json.loads(x)
        for x in (root / "vocabulary-transfer-inputs-01/inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    lookup = {(x["variant"], x["base_id"]): x for x in old}
    rows = []
    for x in old:
        regime, family = x["variant"].split("-")
        if family == "0":
            rows.extend(make_cells(x, lookup[(regime + "-4", x["base_id"])]))
    rows = sorted(rows, key=lambda x: VARIANTS.index(x["variant"]))
    rebuilt = hashlib.sha256("".join(json.dumps(x) + "\r\n" for x in rows).encode()).hexdigest()
    if rebuilt != sha(data / "inputs.jsonl"):
        raise ValueError("input reconstruction mismatch")
    out = Path("docs/pretrained_reuse/R2_TRANSFER_FACTORIAL_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=r,
            factorial=f,
            protocol=p,
            manifest=read(data / "manifest.json"),
            rebuilt_input_sha256=rebuilt,
            hashes={q.name: sha(q) for q in run.iterdir() if q.is_file()},
        ),
    )
    lines = [
        "# Question wording / alias factorial results",
        "",
        "All8192 rows per model completed. No training, calibration or checkpoint selection.",
        "Same128 states, two numeric regimes, four question cells. All targets unchanged.",
        "Generic/domain_alias reference cells reproduce prior probabilities exactly (maxdelta0).",
        "Committed constructor also regenerates the complete prepared input file hash exactly.",
        "",
        "| Model | Cell | Acc | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, variants in r["metrics"].items():
        for cell, m in variants.items():
            m = m["overall"]
            lines.append(
                f"| {name} | {cell} | {m['accuracy']:.6f} | {m['nll']:.6f} |"
                f" {m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += [
        "",
        "## Paired effects",
        "",
        "Difference is second minus first. NegativeAccuracy is worse; positiveNLL/Brier is worse.",
        "| Model | Contrast | Metric | Difference |95%interval|",
        "|---|---|---|---:|---|",
    ]
    for name, contrasts in f["paired_effects"].items():
        for contrast, v in contrasts.items():
            for metric, m in v["metrics"].items():
                lo, hi = m["percentile_95_interval"]
                lines.append(
                    f"| {name} | {contrast} | {metric} | {m['difference']:+.6f} |"
                    f" [{lo:+.6f},{hi:+.6f}] |"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        "For the coverage checkpoint, adding the exact alias preamble to the unchanged",
        "generic question reduces baseAccuracy84.38% to53.03%:−31.35pp,",
        "paired95%interval[−33.98,−28.71]pp. Expanded-range reduction is−24.71pp.",
        "The old checkpoint shows the same failure direction. This is an actual",
        "question-prefix intervention with states/options/targets fixed, not a state-key change.",
        "It does not isolate added token length, distractor field mentions or alias semantics.",
        "",
        "Removing the prefix from the domain family improves40.14% to48.83% in base",
        "and36.13% to40.92% in expanded range; both remain weak. Domain wording without",
        "prefix is still35.55pp below generic in base. Wording changes include syntax as",
        "well as the adjective, so a single-word causal claim remains unjustified.",
        "",
        "The two weaknesses coexist. Do not simply strip the prefix and claim general",
        "reasoning is fixed, or tune a template-specific preprocessor on these labels.",
        "A next learning hypothesis should address train-only compositional instruction",
        "binding with independent templates and retain general original-text validation.",
        "This diagnostic itself launches no such run. Existing models remain preserved.",
        "",
        "## Accounting and limits",
        "",
        "Count/conditional questions are identical retention controls in all four cells.",
        "Full per-task metrics and opposite-condition paired correctness are in JSON.",
        "Repeated states/cells are not independent samples. Paired state bootstrap:",
        "10000draws,seed17,percentile95%, no multiplicity correction. Development data",
        "were observed before this diagnostic; these are not blind audit results.",
        f"Timed inference{r['seconds']:.3f}s, peak allocatedVRAM{r['peak_vram_bytes']}bytes.",
        "Cloud charges0; electricity unmeasured. GPU was idle before launch.",
        "Execution source63b4226; input preparation preceded that commit, and its original",
        "builder hash is retained. Exact reconstruction includes the original Windows CRLF bytes.",
        f"Artifacts `{run.as_posix()}` and `{data.as_posix()}`.",
        "Four modalities>=95%/<=200M objective remains unmet. No image/audio/video inference.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
