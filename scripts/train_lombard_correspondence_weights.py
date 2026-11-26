"""Freeze TRAIN-only scalar readout before calibration/development; no neural fitting."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from scripts.av_slot_weight_metrics import eligible, paired_comparison
from scripts.av_slot_weights import agreement_terms, fit_coefficients, weighted_scores
from scripts.lombard_correspondence import (
    OPTION_IDS,
    build_quartets,
    choice_json,
    choice_logps,
    choice_metrics,
    query_indices,
    select_query_temperature,
    signature,
)
from scripts.lombard_grid_corpus import SLOTS
from scripts.lombard_readout_data import load_data, read_json
from scripts.replay_native_av_correspondence import sha, verify

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PARENT = ROOT / "lombard-correspondence-02"
PARENT_SHA = "9638d06b7fe4c0e3c53e0a0080d38b350eb135fd8cf819e2e3fcfdbf8e977516"
CODE = [
    "scripts/train_lombard_correspondence_weights.py",
    "scripts/av_slot_weights.py",
    "scripts/av_slot_weight_metrics.py",
    "scripts/replay_native_av_correspondence.py",
    "tests/test_av_slot_weights.py",
    "docs/pretrained_reuse/R2_AV_SLOT_WEIGHTS_PROTOCOL_20261015.md",
]
POLICIES = ["random", "hard"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if (
        bool(args.replay_reference) != bool(args.replay_sha256)
        or args.smoke
        and args.replay_reference
    ):
        raise ValueError("invalid frozen execution mode")
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        smoke=args.smoke,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        base_model_updates=0,
        gpu_used=False,
        encoders_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        numpy=np.__version__,
        python=sys.version,
        device="cpu",
        dtype="float64",
        processor=platform.processor(),
        calibration_evaluated=False,
        validation_evaluated=False,
    )

    def guard():
        peak = getattr(
            psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
        )
        size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
        if (
            time.perf_counter() - tick > 600
            or peak > 2 * 2**30
            or size + 2**22 > 64 * 2**20
            or shutil.disk_usage(args.output).free < 2**30
        ):
            raise RuntimeError("scalar readout CPU/time/output/disk guard")

    try:
        verify({str(PARENT / "result.json"): PARENT_SHA})
        parent = read_json(PARENT / "result.json")
        if parent["status"] != "PASS_FROZEN_NATIVE_CORRESPONDENCE":
            raise ValueError("complete parent required")
        pins = dict(parent["validated_hashes"])
        pins[str(PARENT / "result.json")] = PARENT_SHA
        pins.update({str(PARENT / k): h for k, h in parent["artifact_hashes"].items()})
        pins.update({str(Path(p).resolve()): sha(p) for p in CODE})
        for p in Path(psutil.__file__).resolve().parent.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".pyd", ".dll"}:
                pins[str(p.resolve())] = sha(p)
        verify(pins)
        guard()
        _, _, rows, provenance = load_data(ROOT, smoke=False)
        if any(parent["provenance"][k] != v for k, v in provenance.items()):
            raise ValueError("source/split provenance differs")
        result.update(
            validated_hashes=pins,
            provenance=provenance,
            slot_order=list(SLOTS),
            registered_parameters_and_coefficients=122790905,
            fit_parameter_count=6,
        )
        catalogs = {p: {} for p in POLICIES}
        indices = {p: {} for p in POLICIES}
        counts = {p: {} for p in POLICIES}
        for p in POLICIES:
            for split in rows:
                q, c = build_quartets(rows[split], p)
                old = np.load(PARENT / f"{p}-{split}-catalog.npy", allow_pickle=False)
                if not np.array_equal(q, old):
                    raise ValueError("original frozen catalogs differ")
                catalogs[p][split] = q
                indices[p][split] = query_indices(q)
                counts[p][split] = c
        result["catalog_counts"] = counts

        def load_terms(split):
            logits = {}
            for arm in ["audio", "video_ordered"]:
                name = dict(
                    train="best-train.npz", calibration="calibration.npz", validation="best-raw.npz"
                )[split]
                with np.load(
                    ROOT / "lombard-temporal-full-01" / arm / name, allow_pickle=False
                ) as saved:
                    if saved["source_ids"].tolist() != [r["source_id"] for r in rows[split]]:
                        raise ValueError("native raw-logit IDs differ")
                    logits[arm] = saved["logits"].copy()
            values = {}
            for p in POLICIES:
                query, videos, _ = indices[p][split]
                values[p] = agreement_terms(
                    logits["audio"][query],
                    logits["video_ordered"][videos],
                    parent["models"]["audio"]["temperatures"],
                    parent["models"]["video_ordered"]["temperatures"],
                )
                with np.load(PARENT / f"{p}-{split}.npz", allow_pickle=False) as reference:
                    if not np.array_equal(
                        weighted_scores(values[p], np.ones(6)), reference["scores"]
                    ):
                        raise ValueError("all1 baseline score is not exact")
            return values

        train = load_terms("train")
        x = np.concatenate([train[p] for p in POLICIES])
        y = np.concatenate([indices[p]["train"][2] for p in POLICIES])
        if len(x) != 32640:
            raise ValueError("complete TRAIN roles required")
        contexts = set()
        for p in POLICIES:
            q, v, _ = indices[p]["train"]
            contexts.update((int(a), tuple(sorted(b.tolist()))) for a, b in zip(q, v, strict=True))
        result["training_accounting"] = dict(
            full_batch_roles=len(x),
            available_parent_assets=4080,
            unique_query_contexts=len(contexts),
            repeated_contexts_per_batch=len(x) - len(contexts),
            actual_train_rows_used=8 if args.smoke else len(x),
        )
        used = 8 if args.smoke else len(x)
        ordered_queries = np.concatenate([indices[p]["train"][0] for p in POLICIES])[:used]
        ordered_candidates = np.concatenate([indices[p]["train"][1] for p in POLICIES])[:used]
        result["training_accounting"].update(
            actual_unique_query_assets=len(set(ordered_queries.tolist())),
            actual_unique_candidate_assets=len(set(ordered_candidates.ravel().tolist())),
            actual_unique_parent_assets=len(
                set(ordered_queries.tolist()) | set(ordered_candidates.ravel().tolist())
            ),
            actual_unique_query_contexts=len(
                {
                    (int(a), tuple(sorted(b.tolist())))
                    for a, b in zip(ordered_queries, ordered_candidates, strict=True)
                }
            ),
            sample_order_sha256=hashlib.sha256(
                ordered_queries.tobytes() + ordered_candidates.tobytes()
            ).hexdigest(),
        )
        if args.smoke:
            x, y = x[:8], y[:8]
        fit_tick = time.perf_counter()
        weights, history = fit_coefficients(x, y, steps=8 if args.smoke else 512, monitor=guard)
        result.update(
            fit_seconds=time.perf_counter() - fit_tick,
            iterations=len(history) - 1,
            accepted_iterations=sum(h["accepted"] for h in history),
            changed_iterations=sum(h["changed"] for h in history),
            iteration_role_presentations=(len(history) - 1) * len(x),
            objective_gradient_evaluations=1
            + sum(h["backtracks"] + 1 if h["accepted"] else 24 for h in history[1:]),
            objective_label_presentations=len(x)
            * (1 + sum(h["backtracks"] + 1 if h["accepted"] else 24 for h in history[1:])),
            weights=weights.tolist(),
        )
        if any(a["nll"] < b["nll"] - 1e-12 for a, b in zip(history, history[1:], strict=False)):
            raise ValueError("TRAIN NLL increased")
        wp = args.output / "weights.npy"
        np.save(wp, weights)
        if not np.array_equal(np.load(wp, allow_pickle=False), weights):
            raise ValueError("coefficient reload differs")
        hp = args.output / "training-history.jsonl"
        hp.write_text("".join(json.dumps(h) + "\n" for h in history), encoding="utf-8")
        result.update(
            weights_sha256=sha(wp),
            weights_frozen_before_calibration=True,
            weights_frozen_utc=datetime.now(UTC).isoformat(),
            training_history_sha256=sha(hp),
        )
        curves = {}
        for h in history:
            if h["iteration"] in [0, 1, 32, 64, 128, 256, 384, 512]:
                curves[str(h["iteration"])] = {
                    p: choice_metrics(
                        np.exp(choice_logps(weighted_scores(train[p], h["weights"]))),
                        indices[p]["train"][2],
                        choice_logps(weighted_scores(train[p], h["weights"])),
                    )
                    for p in POLICIES
                }
        result["train_curves"] = curves
        if not args.smoke:
            cal = load_terms("calibration")
            cal_scores = np.concatenate([weighted_scores(cal[p], weights) for p in POLICIES])
            cal_y = np.concatenate([indices[p]["calibration"][2] for p in POLICIES])
            temperature, calibration = select_query_temperature(cal_scores, cal_y)
            frozen = dict(
                weights_sha256=sha(wp),
                weights=weights.tolist(),
                query_temperature=temperature,
                calibration=calibration,
                selection="fixed512 TRAIN-only; calibration-only scalar temperature",
            )
            fp = args.output / "frozen-readout.json"
            fp.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
            result.update(
                query_temperature=temperature,
                calibration=calibration,
                calibration_evaluated=True,
                frozen_readout_sha256=sha(fp),
                readout_frozen_before_validation=True,
                readout_frozen_utc=datetime.now(UTC).isoformat(),
            )
            validation = load_terms("validation")
            result["validation_evaluated"] = True
            result["policies"] = {}
            for p in POLICIES:
                report = dict(raw={}, calibrated={})
                for split, terms in [
                    ("train", train[p]),
                    ("calibration", cal[p]),
                    ("validation", validation[p]),
                ]:
                    score = weighted_scores(terms, weights)
                    logp = choice_logps(score, temperature)
                    probs = np.exp(logp)
                    query, videos, target = indices[p][split]
                    report["raw"][split] = choice_metrics(
                        np.exp(choice_logps(score)), target, choice_logps(score)
                    )
                    report["calibrated"][split] = choice_metrics(probs, target, logp)
                    np.savez_compressed(
                        args.output / f"{p}-{split}.npz",
                        scores=score,
                        probabilities=probs,
                        log_probabilities=logp,
                        targets=target,
                        query_ids=np.array([rows[split][i]["source_id"] for i in query]),
                        candidate_ids=np.array(
                            [[rows[split][i]["source_id"] for i in r] for r in videos]
                        ),
                    )
                query, videos, target = indices[p]["validation"]
                score = weighted_scores(validation[p], weights)
                logp = choice_logps(score, temperature)
                probs = np.exp(logp)
                with np.load(PARENT / f"{p}-validation.npz", allow_pickle=False) as saved:
                    baseline = saved["probabilities"].copy()
                    baseline_logp = saved["log_probabilities"].copy()
                report["baseline"] = parent["policies"][p]["calibrated"]["validation"]
                report["candidate_minus_baseline"] = paired_comparison(
                    probs,
                    logp,
                    baseline,
                    baseline_logp,
                    target,
                    [rows["validation"][i]["speaker_group"] for i in query],
                )
                labels = {
                    "condition": np.array([rows["validation"][i]["condition"] for i in query])
                }
                sig = [signature(r) for r in rows["validation"]]
                labels["closest_negative_distance"] = np.array(
                    [
                        min(
                            sum(a != b for a, b in zip(sig[q], sig[v], strict=True))
                            for v in vs
                            if v != q
                        )
                        for q, vs in zip(query, videos, strict=True)
                    ]
                )
                strata = []
                for category, values in labels.items():
                    for value in np.unique(values):
                        mask = values == value
                        c = choice_metrics(probs[mask], target[mask], logp[mask])
                        b = choice_metrics(baseline[mask], target[mask], baseline_logp[mask])
                        strata.append(
                            dict(
                                category=category,
                                value=str(value),
                                count=int(mask.sum()),
                                candidate=c,
                                baseline=b,
                                accuracy_delta=c["accuracy"] - b["accuracy"],
                            )
                        )
                report["strata"] = strata
                report["controls"] = {}
                matrix = score.reshape(-1, 4, 4)
                for name, s, corrected in [
                    ("audio_absent", np.zeros_like(score), None),
                    ("video_absent", np.zeros_like(score), None),
                    ("audio_swap", np.roll(matrix, -1, axis=1).reshape(-1, 4), (target + 1) % 4),
                    ("video_swap", np.roll(matrix, -1, axis=2).reshape(-1, 4), (target - 1) % 4),
                ]:
                    lp = choice_logps(s, temperature)
                    pp = np.exp(lp)
                    entry = dict(original=choice_metrics(pp, target, lp))
                    if corrected is not None:
                        entry["corrected"] = choice_metrics(pp, corrected, lp)
                        if any(
                            not np.isclose(
                                entry["corrected"][k],
                                report["calibrated"]["validation"][k],
                                atol=1e-12,
                                rtol=0,
                            )
                            for k in ["accuracy", "nll", "brier", "ece"]
                        ):
                            raise ValueError("counterfactual corrected metrics differ")
                    report["controls"][name] = entry
                    np.savez_compressed(
                        args.output / f"{p}-{name}.npz",
                        scores=s,
                        probabilities=pp,
                        log_probabilities=lp,
                        targets=target,
                        corrected_targets=target if corrected is None else corrected,
                        query_ids=np.array(
                            [
                                rows["validation"][i]["source_id"]
                                for i in (
                                    np.roll(catalogs[p]["validation"], -1, axis=1).ravel()
                                    if name == "audio_swap"
                                    else query
                                )
                            ]
                        ),
                        candidate_ids=np.array(
                            [
                                [rows["validation"][i]["source_id"] for i in v]
                                for v in (
                                    np.roll(videos, -1, axis=1) if name == "video_swap" else videos
                                )
                            ]
                        ),
                    )
                for s in score:
                    if choice_json(s, OPTION_IDS, temperature) != choice_json(
                        s[::-1], OPTION_IDS[::-1], temperature
                    ):
                        raise ValueError("candidate-ID reorder differs")
                report["json_option_permutation_exact"] = True
                report["eligible_for_runtime_trial"] = eligible(
                    report["calibrated"]["validation"], report["baseline"], strata
                )
                result["policies"][p] = report
            result["eligible_for_runtime_trial"] = all(
                v["eligible_for_runtime_trial"] for v in result["policies"].values()
            )
            if sha(wp) != frozen["weights_sha256"] or sha(fp) != result["frozen_readout_sha256"]:
                raise ValueError("readout changed during evaluation")
        verify(pins)
        guard()
        result.update(
            status="PASS_TRAIN_ONLY_SLOT_WEIGHT_SMOKE"
            if args.smoke
            else "PASS_FIXED_SLOT_WEIGHT_COMPARISON",
            source_and_base_weights_unchanged=True,
        )
        if args.replay_reference:
            prior = read_json(args.replay_reference / "result.json", args.replay_sha256)
            if (
                prior["status"] != "PASS_FIXED_SLOT_WEIGHT_COMPARISON"
                or prior["validated_hashes"] != pins
            ):
                raise ValueError("complete same-code primary required")
            verify({str(args.replay_reference / k): h for k, h in prior["artifact_hashes"].items()})
            for k in [
                "weights",
                "query_temperature",
                "calibration",
                "training_accounting",
                "train_curves",
                "policies",
                "eligible_for_runtime_trial",
            ]:
                if result[k] != prior[k]:
                    raise ValueError("fresh TRAIN-only fit/report differs: " + k)
            for name in prior["artifact_hashes"]:
                a, b = args.output / name, args.replay_reference / name
                if name.endswith(".npz"):
                    with (
                        np.load(a, allow_pickle=False) as current,
                        np.load(b, allow_pickle=False) as old,
                    ):
                        if set(current.files) != set(old.files) or any(
                            not np.array_equal(current[k], old[k]) for k in current.files
                        ):
                            raise ValueError("fresh prediction payload differs")
                elif sha(a) != sha(b):
                    raise ValueError("fresh weights/history/frozen config differs")
            verify(pins)
            verify({str(args.replay_reference / "result.json"): args.replay_sha256})
        result.update(
            fresh_process_replay=bool(args.replay_reference),
            artifact_hashes={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - tick,
            ended_utc=datetime.now(UTC).isoformat(),
            peak_cpu_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
        )
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
