"""Publish actual native execution separately from task-quality qualification."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.replay_native_av_correspondence import sha, verified_primary, verify

PREFIX = "R5_NATIVE_AV_CORRESPONDENCE_RESULTS_20261015"


def admit(primary, replay, strict):
    if (
        any(
            r["status"] != "PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME"
            or r["model_updates"] != 0
            or r["gpu_used"]
            or r["sealed_audit_opened"]
            for r in [primary, replay]
        )
        or not replay["fresh_process_replay"]
        or primary["profile_sha256"] != replay["profile_sha256"]
        or strict["status"] != "PASS_STRICT_RAW_AV_REPLAY_PROVENANCE"
        or not strict["profile_checked_before_forward"]
        or strict["expected_profile_sha256"] != primary["profile_sha256"]
    ):
        raise ValueError("complete unchanged profile-bound raw execution required")


def main():
    parser = argparse.ArgumentParser()
    for label in ["primary", "replay"]:
        parser.add_argument("--" + label, type=Path, required=True)
        parser.add_argument("--" + label + "-sha256", required=True)
    parser.add_argument("--strict-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    first, pins = verified_primary(args.primary, args.primary_sha256)
    replay, more = verified_primary(args.replay, args.replay_sha256)
    for p, h in more.items():
        if p in pins and pins[p] != h:
            raise ValueError("conflicting source authorities")
        pins[p] = h
    certificate = args.replay / "replay-provenance.json"
    pins[str(certificate)] = args.strict_sha256
    pins[str(Path(__file__).resolve())] = sha(__file__)
    verify(pins)
    strict = json.loads(certificate.read_text(encoding="utf-8"))
    if strict["replay_receipt_sha256"] != args.replay_sha256:
        raise ValueError("strict certificate binds another replay")
    for p, h in strict["validated_hashes"].items():
        if p in pins and pins[p] != h:
            raise ValueError("strict source authority conflicts")
        pins[p] = h
    verify(pins)
    admit(first, replay, strict)
    for k in [
        "selected_source_ids",
        "quartets",
        "decisions",
        "feature_parity",
        "backend_calls",
        "cache_hits",
        "cache_entries",
        "parameters_and_coefficients",
    ]:
        if first[k] != replay[k]:
            raise ValueError("full runtime report parity differs")
    with (
        np.load(args.primary / "native-states.npz", allow_pickle=False) as a,
        np.load(args.replay / "native-states.npz", allow_pickle=False) as b,
    ):
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("full state arrays differ")
    workers = [
        json.loads((p / "video-worker/result.json").read_text(encoding="utf-8"))
        for p in [args.primary, args.replay]
    ]
    fields = [
        "source_commit",
        "command",
        "started_utc",
        "ended_utc",
        "seconds",
        "selected_source_ids",
        "selection",
        "quartets",
        "decisions",
        "feature_parity",
        "backend_calls",
        "cache_hits",
        "cache_entries",
        "audio_load_seconds",
        "video_load_seconds",
        "decision_p50_seconds",
        "decision_p95_seconds",
        "observed_sampled_combined_ram_peak",
        "peak_cpu_bytes",
        "profile_sha256",
        "parameters_and_coefficients",
        "python",
        "torch",
        "numpy",
        "device",
        "neural_dtype",
        "scoring_dtype",
        "scope",
    ]
    small = dict(
        status="VERIFIED_NATIVE_AV_FUNCTIONAL_PARITY_NOT_OMNI_COMPLETION",
        primary={k: first[k] for k in fields},
        replay={k: replay[k] for k in fields},
        receipt_hashes={
            str(args.primary / "result.json"): args.primary_sha256,
            str(args.replay / "result.json"): args.replay_sha256,
            str(certificate): args.strict_sha256,
        },
        strict_wrapper={
            k: strict[k]
            for k in [
                "source_commit",
                "command",
                "seconds",
                "started_utc",
                "ended_utc",
                "input_pin_count",
                "profile_checked_before_forward",
            ]
        },
        workers=[
            {
                k: w[k]
                for k in [
                    "parameters",
                    "source_revision",
                    "parameter_sha_before",
                    "parameter_sha_after",
                    "python",
                    "jax",
                    "numpy",
                    "tensorflow",
                    "av",
                    "load_seconds",
                    "seconds",
                    "peak_cpu_bytes",
                    "calls",
                ]
            }
            for w in workers
        ],
        input_pin_count=len(pins),
        state_arrays_exact=True,
        custom_json_exact=True,
        zero_selected_id_changes=True,
        model_updates=0,
        gpu_used=False,
        cloud_cost=0,
        remaining=[
            "independent/OOD quality",
            "complete four-modality runtime",
            "task95–98%",
            "full package/latency qualification",
        ],
    )
    docs = Path("docs/pretrained_reuse")
    jp = docs / (PREFIX + ".json")
    mp = docs / (PREFIX + ".md")
    jp.write_text(json.dumps(small, indent=2) + "\n", encoding="utf-8")
    n = len(first["selected_source_ids"])
    pdelta = max(r["max_abs_probability_delta"] for r in first["decisions"])
    adelta = max(r["audio_max_abs_feature_delta"] for r in first["feature_parity"])
    lines = [
        "# Actual native AV correspondence and reusable states",
        "",
        "## What actually ran",
        "",
        f"{n} native audio clips and{n} front videos from the first TRAIN anchor’s",
        "frozen random/hard quartets, all four query roles each. Full raw ZIP bytes,",
        "native decoding/frontends/encoders, frozen audio24/video18 heads and custom",
        "four-option JSON completed. No text question/transcript/label as model input.",
        "No training or model changes. All old source/corpora/checkpoints remain intact.",
        "",
        f"Actual backend calls: audio{first['backend_calls']['audio']}, "
        f"video{first['backend_calls']['video']}.",
        f"{first['cache_hits']} repeated raw requests are cache hits;128additional decisions",
        "invoke neither encoder. Profile/media/modality/time keys and bounded LRU",
        "retain immutable owner-bound51-logit states. This is task-specific decision",
        "state reuse, not a general world-state or temporal dynamics model. Metadata",
        "time ranges are local clip ranges, not a newly proved common acquisition clock.",
        "",
        "## Parity, provenance and limitations",
        "",
        "All selected native CPU video frame features and raw logits match the original",
        "native CPU bank exactly. Audio is measured CPU versus the original CUDA bank:",
        f"max pooled-feature difference{adelta:.9g}; max four-choice probability difference",
        f"{pdelta:.9g},0 selected-ID changes. These pass the predeclared tolerances",
        "1e-4(feature)/1e-3(head)/1e-4(probability); no tolerance tuning after execution.",
        "Full separate-process CPU/raw replay is byte-exact for native features/logits",
        "and all8 JSON decisions; legitimate custom-ID/video reordering is exact.",
        "",
        "Review found a worker dependency replay gap in the immutable producer.",
        "An independent strict wrapper binds physical dependencies in both processes",
        "before forwards, requires the complete backend profile SHA and rechecks",
        "afterwards. Its separate receipt binds the completed numerical replay. No",
        "executed producer/backend/scoring source was changed to pass these gates.",
        "",
        "This is a TRAIN functional pilot, not independent quality evidence. The",
        "preceding observed development results remain94.85% random /84.08% near-command",
        "correspondence, with their previously recorded uncertainties. No new blind",
        "result, per-modality95–98%, natural image/video reasoning, optional condition",
        "generality, sensor quality or complete Omni release is claimed.",
        "",
        "## Actual CPU resources",
        "",
        "| Execution | Seconds (producer) | Parent peak GiB | Worker peak GiB | "
        "Sampled concurrent GiB | Score-only p50 / p95 ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, r, w in zip(["primary", "fresh replay"], [first, replay], workers, strict=True):
        lines.append(
            f"| {label} | {r['seconds']:.3f} | {r['peak_cpu_bytes'] / 2**30:.3f} | "
            f"{w['peak_cpu_bytes'] / 2**30:.3f} | "
            f"{r['observed_sampled_combined_ram_peak'] / 2**30:.3f} | "
            f"{r['decision_p50_seconds'] * 1e3:.4f} / "
            f"{r['decision_p95_seconds'] * 1e3:.4f} |"
        )
    lines += [
        "",
        "Score-only latency excludes cold load, hashes, ZIP/decode, preprocessing,",
        "native encoder and head forward. It is not end-to-end raw latency. Raw",
        "encoding and child decode/forward/cold timings are in the small JSON/receipts.",
        "Sampled concurrent RAM includes only this producer and its own children;",
        "individual peaks are separate-process maxima. Windows approved system Python",
        "uses the existing isolated JAX/TF dependency libraries; Torch remains in its",
        "existing environment. No venv-executable policy override/dependency upgrade.",
        "CPU i7-12700H; FP32 neural/FP64 scoring. Zero GPU/cloud/gradient/model updates.",
        "",
        "122790899 parameters/coefficients count both full106355 heads, native",
        "encoders122573568, two2304 normalizers,12native temperatures and1query",
        "temperature. No dead-branch pruning/deduplication credit. This AV composition",
        "does not count/prove a complete four-modality product, package size or phone",
        "runtime. Weights are referenced locally; this is not a relocated release bundle.",
        "",
        f"[Exact commands, runtime, resource/feature deltas and receipt hashes]({PREFIX}.json).",
        "Source/native/learned/media/runtime/output inputs are verified before/after",
        "publication; large raw media and weight/checkpoint files stay outside Git.",
        "",
        "Next remains independent task/OOD and conditional-semantic quality coverage",
        "before sharing/merging/compression or product adoption. Current goal is active",
        "and unmet; preserve this runnable AV primitive as a measured reference.",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    verify(pins)
    receipt = dict(
        status="PASS_HASH_BOUND_NATIVE_AV_PUBLICATION",
        validated_hashes=pins,
        input_pin_count=len(pins),
        output_hashes={str(p.resolve()): sha(p) for p in [jp, mp]},
    )
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["status"])


if __name__ == "__main__":
    main()
