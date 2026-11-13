"""Publish full hash-bound correspondence receipts and exact prediction replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

KEYS = ["catalog_counts", "models", "query_temperature", "calibration", "policies"]
PREFIX = "R2_LOMBARD_CORRESPONDENCE_RESULTS_20261015"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def verify(pins):
    if not pins or any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("publication source/report/prediction hash differs")


def merge_pins(pins, given):
    if any(path in pins and pins[path] != expected for path, expected in given.items()):
        raise ValueError("conflicting source authorities")
    pins.update(given)


def main():
    parser = argparse.ArgumentParser()
    for label in ["primary", "replay"]:
        parser.add_argument("--" + label, type=Path, required=True)
        parser.add_argument("--" + label + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    folders = [args.primary, args.replay]
    hashes = [args.primary_sha256, args.replay_sha256]
    pins = {str(p / "result.json"): h for p, h in zip(folders, hashes, strict=True)}
    verify(pins)
    reports = [json.loads((p / "result.json").read_text()) for p in folders]
    first, replay = reports
    if (
        any(r["status"] != "PASS_FROZEN_NATIVE_CORRESPONDENCE" for r in reports)
        or not replay["fresh_process_replay"]
        or first["validated_hashes"] != replay["validated_hashes"]
        or any(first[k] != replay[k] for k in KEYS)
    ):
        raise ValueError("full unchanged fresh replay required")
    for folder, report in zip(folders, reports, strict=True):
        merge_pins(pins, report["validated_hashes"])
        merge_pins(pins, {str(folder / name): h for name, h in report["artifact_hashes"].items()})
    merge_pins(pins, {str(Path(__file__).resolve()): sha(__file__)})
    verify(pins)
    if set(first["artifact_hashes"]) != set(replay["artifact_hashes"]):
        raise ValueError("full prediction file set differs")
    for name in first["artifact_hashes"]:
        a, b = (folder / name for folder in folders)
        if name.endswith(".npz"):
            with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
                if set(x.files) != set(y.files) or any(
                    not np.array_equal(x[k], y[k]) for k in x.files
                ):
                    raise ValueError("full prediction arrays differ")
        elif name.endswith(".npy"):
            if not np.array_equal(np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)):
                raise ValueError("catalog array differs")
        elif sha(a) != sha(b):
            raise ValueError("custom JSON differs")
    result = {
        k: first[k]
        for k in KEYS
        + [
            "provenance",
            "source_commit",
            "composition_parameters_and_coefficients",
            "scoring",
            "scope",
        ]
    }
    result.update(
        status="VERIFIED_NATIVE_CORRESPONDENCE_NOT_PRODUCT_COMPLETION",
        input_pin_count=len(pins),
        receipt_hashes=dict(zip(map(str, folders), hashes, strict=True)),
        artifact_hashes={
            str(p): r["artifact_hashes"] for p, r in zip(folders, reports, strict=True)
        },
        execution={
            str(p): {
                k: r[k]
                for k in [
                    "command",
                    "started_utc",
                    "ended_utc",
                    "seconds",
                    "peak_cpu_bytes",
                    "output_bytes",
                    "python",
                    "torch",
                    "numpy",
                    "device",
                    "neural_dtype",
                    "scoring_dtype",
                    "threads",
                    "interop_threads",
                    "deterministic",
                ]
            }
            for p, r in zip(folders, reports, strict=True)
        },
        exact_full_payload_replay=True,
        model_updates=0,
        gradients_computed=False,
        gpu_used=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        next_gate="Independent task/OOD coverage; no development tuning or promotion",
    )
    docs = Path("docs/pretrained_reuse")
    small = docs / (PREFIX + ".json")
    small.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Native audio and four-video correspondence",
        "",
        "## Measured result",
        "",
        "Frozen audio24 and video_ordered18 heads perform a new controlled four-choice",
        "native correspondence task. Same-speaker/recording-condition candidates remove",
        "speaker/background as a matching cue. Every quartet supplies all four audio",
        "query roles with fixed videos; each candidate is correct exactly once.",
        "",
        "| Policy / split | Accuracy | NLL | Brier | ECE | Mean confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for policy, report in first["policies"].items():
        for split, m in report["calibrated"].items():
            lines.append(
                f"| {policy} / {split} | {m['accuracy']:.2%} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece']:.6f} | {m['mean_confidence']:.2%} |"
            )
    lines += [
        "",
        "Calibration-only search chose query temperature1.0 from the fixed17-point",
        "grid on3184 calibration roles; raw and calibrated outputs are therefore equal.",
        "Random94.85% falls to84.08% with near-command negatives. This is a task",
        "difficulty difference, not paired model improvement or a new blind result.",
        "Train/dev gaps are4.15 and10.18 percentage points respectively; the hard task",
        "shows substantially weaker transfer. Do not replace it with the easier score.",
        "",
        "## Fixed evidence controls",
        "",
        "| Policy / intervention | Original Accuracy | Original NLL | Corrected Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for policy, report in first["policies"].items():
        for control, row in report["controls"].items():
            m = row["original_targets"]
            corrected = (
                f"{row['corrected_targets']['accuracy']:.2%}"
                if "corrected_targets" in row
                else "N/A"
            )
            lines.append(
                f"| {policy} / {control} | {m['accuracy']:.2%} | {m['nll']:.6f} | {corrected} |"
            )
    lines += [
        "",
        "Missing inputs use an explicit uniform25% prototype policy. This comparison",
        "does not prove superiority over unrestricted trained unimodal models. Videos",
        "alone are identical across four balanced query roles, limiting a deterministic",
        "video-only answer to25%. Audio-only lacks the opaque candidate-video mapping.",
        "Content swaps under unchanged IDs invalidate original labels; corrected targets",
        "restore matched metrics. Legitimate option+ID reorder is exactly stable on all",
        "7112 development query roles. The fixed matching function sees posterior",
        "arrays only, never source identities, ground-truth labels or transcript text.",
        "This supports use of both streams in this task, not open-world Omni quality.",
        "",
        "## Grouped uncertainty and accounting",
        "",
        "| Policy | Accuracy minus uniform | Speaker bootstrap95% interval |",
        "| --- | --- | --- |",
    ]
    for policy, report in first["policies"].items():
        d = report["matched_minus_uniform"]["differences"]["accuracy"]
        lines.append(f"| {policy} | {d['point']:.2%} | [{d['lower']:.2%}, {d['upper']:.2%}] |")
    lines += [
        "",
        "Whole-speaker paired bootstrap10000/seed17/linear percentile95%; all repeated",
        "quartet roles retained within nine clusters. NLL/Brier/ECE intervals in JSON.",
        "3556 roles per policy derive from889 assets, not3556 independent new sources.",
        "Random889 unique unordered quartets; hard874,3496 unique query contexts and60",
        "repeated contexts. All889 parents retained. No train/cal/dev media/source",
        "overlap; hard distance refers to six-slot Hamming distance, not a phonetic",
        "LibriSpeech benchmark. Original51 semantic options and25 letters unchanged.",
        "4080 training assets/398 calibration assets remain; no new training updates.",
        "",
        "## Reproducibility, resources and limits",
        "",
        "Primary01 succeeded; replay01 failed at integer-key versus serialized string-key",
        "report comparison. All attempts and original execution source remain intact.",
        "Version2 only fixes finite JSON-domain equality, with no rounding/tolerance.",
        "Primary02 and separate replay02 complete all exact prediction/catalog/JSON,",
        "full model reload and terminal hash checks. No performance-driven change.",
        "",
        "| Execution | Seconds | Peak RAM GiB | Output MB | Receipt SHA256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for folder, h, r in zip(folders, hashes, reports, strict=True):
        lines.append(
            f"| {folder.name} | {r['seconds']:.3f} | {r['peak_cpu_bytes'] / 2**30:.3f} | "
            f"{r['output_bytes'] / 1e6:.3f} | `{h}` |"
        )
    lines += [
        "",
        f"Execution source:`{first['source_commit']}`. Exact commands/timestamps/runtime,",
        "all split/checkpoint/catalog hashes, full raw/calibrated/control metrics and",
        f"[small receipt]({PREFIX}.json) are retained. CPU i7-12700H,4threads/interOp1,",
        "FP32 neural/FP64 scoring, deterministic; zero GPU/cloud/encoder loads,",
        "gradients or model updates. Existing runs/configs/checkpoints remain unchanged.",
        "Composition122790899 registered parameters/coefficients includes native encoders,",
        "two full106355 heads, both2304 normalizers,12 native temperatures and1 query",
        "temperature. No pruning credit; AV slice only, not complete four-modality size.",
        "",
        "This reuses selected observed development data and structured spoken commands.",
        "No sealed audit, general-purpose temporal/causal reasoning, sensor fusion,",
        "per-modality95–98%, complete resident package/latency or product adoption claim.",
        "Keep this as an actual evidence-dependent AV baseline; next requires independent",
        "task/OOD coverage rather than tuning these negatives/temperatures/checkpoints.",
        "",
    ]
    md = docs / (PREFIX + ".md")
    md.write_text("\n".join(lines), encoding="utf-8")
    verify(pins)
    receipt = dict(
        status="PASS_HASH_BOUND_CORRESPONDENCE_PUBLICATION",
        validated_hashes=pins,
        input_pin_count=len(pins),
        output_hashes={str(p.resolve()): sha(p) for p in [small, md]},
    )
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["status"])


if __name__ == "__main__":
    main()
