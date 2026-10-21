"""Frozen native head reuse for four-video correspondence; no neural model fitting."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_lombard_temporal import audit_arm
from scripts.lombard_correspondence import (
    OPTION_IDS,
    agreement_scores,
    bootstrap_minus_uniform,
    build_quartets,
    choice_json,
    choice_logps,
    choice_metrics,
    choice_probabilities,
    query_indices,
    select_query_temperature,
)
from scripts.lombard_readout_data import read_json, sha
from scripts.lombard_temporal_full import verify_snapshot
from scripts.robomnist_audio_cue import recheck
from scripts.train_lombard_temporal import ROOT, bound_sources, guard, source_scope, state_sha

TRAINING = ROOT / "lombard-temporal-full-01"
EVALUATION = ROOT / "lombard-temporal-evaluation-01"
FIT_SHA = "09ef033c461313a84d34a7db5b6a754c57619871ba1ea9e7f9bdb746e830ec0e"
AUDIT_SHA = "c97d975df27f4931be0b256e5a0ed90f476b2ba7dbbc7534085b7667f3694fdc"
POLICIES = ["random", "hard"]
EXTRA = [
    "scripts/lombard_correspondence.py",
    "scripts/evaluate_lombard_correspondence.py",
    "tests/test_lombard_correspondence.py",
    "docs/pretrained_reuse/R2_LOMBARD_CORRESPONDENCE_PROTOCOL_20261015.md",
]


def bounded(output, tick):
    state = guard(output, tick)
    if state["seconds"] > 600 or state["output_bytes"] + 2**22 > 64 * 2**20:
        raise RuntimeError("fixed correspondence diagnostic safety limit exceeded")
    return state


def snapshot(
    output, name, scores, temperature, query, candidate, targets, parent_rows, corrected=None
):
    probabilities = choice_probabilities(scores, temperature)
    logp = choice_logps(scores, temperature)
    query_ids = np.array([parent_rows[i]["source_id"] for i in query])
    video_ids = np.array([[parent_rows[i]["source_id"] for i in row] for row in candidate])
    path = output / (name + ".npz")
    if path.exists():
        raise ValueError("new output snapshots required")
    np.savez_compressed(
        path,
        scores=scores,
        probabilities=probabilities,
        log_probabilities=logp,
        targets=targets,
        corrected_targets=targets if corrected is None else corrected,
        predictions=probabilities.argmax(1),
        audio_source_ids=query_ids,
        video_source_ids=video_ids,
        speaker_groups=np.array([parent_rows[i]["speaker_group"] for i in query]),
    )
    with np.load(path, allow_pickle=False) as saved:
        if (
            not np.array_equal(scores, saved["scores"])
            or not np.array_equal(probabilities, saved["probabilities"])
            or not np.array_equal(query_ids, saved["audio_source_ids"])
            or not np.array_equal(video_ids, saved["video_source_ids"])
        ):
            raise ValueError("prediction readback differs")
    result = dict(original_targets=choice_metrics(probabilities, targets, logp))
    if corrected is not None:
        result["corrected_targets"] = choice_metrics(probabilities, corrected, logp)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256):
        raise ValueError("replay reference and SHA must both be supplied")
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        policies={},
        models={},
        gradients_computed=False,
        model_updates=0,
        gpu_used=False,
        encoders_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        python=sys.version,
        torch=torch.__version__,
        numpy=np.__version__,
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=subprocess.list2cmdline(
            [sys.executable, "-m", "scripts.evaluate_lombard_correspondence", *sys.argv[1:]]
        ),
        composition_parameters_and_coefficients=122790899,
        scoring="sum of six log full-native-option posterior agreements",
        resource_bounds=dict(
            seconds=600, peak_cpu_bytes=3 * 2**30, output_bytes=64 * 2**20, disk_reserve_bytes=2**30
        ),
        scope="Constructed symmetric correspondence task on observed development; not blind",
    )
    try:
        bounded(args.output, tick)
        fit, audit = (
            read_json(TRAINING / "result.json", FIT_SHA),
            read_json(EVALUATION / "result.json", AUDIT_SHA),
        )
        if (
            fit["status"] != "PASS_FIVE_ARM_TRAINING"
            or audit["status"] != "PASS_FRESH_REPLAY_CONTROLS_AND_GROUPED_COMPARISON"
        ):
            raise ValueError("verified frozen comparison required")
        pins = dict(fit["validated_hashes"])
        if pins != bound_sources():
            raise ValueError("frozen source/runtime differs")
        pins.update(fit["full_execution_hashes"])
        pins.update({str(Path(p).resolve()): sha(Path(p)) for p in EXTRA})
        pins[str((TRAINING / "result.json").resolve())] = FIT_SHA
        pins[str((EVALUATION / "result.json").resolve())] = AUDIT_SHA
        pins[str((TRAINING / "effective-config.json").resolve())] = fit["effective_config_sha256"]
        for arm in ["audio", "video_ordered"]:
            folder = TRAINING / arm
            pins.update(
                {
                    str((folder / k).resolve()): h
                    for k, h in fit["arms"][arm]["artifact_hashes"].items()
                }
            )
            pins[str((folder / "result.json").resolve())] = sha(folder / "result.json")
            if (
                read_json(folder / "result.json", pins[str((folder / "result.json").resolve())])
                != fit["arms"][arm]
            ):
                raise ValueError("parent arm/controller differs")
        recheck(pins)
        reference = None
        if args.replay_reference:
            reference = read_json(args.replay_reference / "result.json", args.replay_sha256)
            if (
                reference["status"] != "PASS_FROZEN_NATIVE_CORRESPONDENCE"
                or reference["validated_hashes"] != pins
            ):
                raise ValueError("complete same-code primary correspondence required")
            recheck(
                {
                    str((args.replay_reference / k).resolve()): h
                    for k, h in reference["artifact_hashes"].items()
                }
            )
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result.update(
            device="cpu",
            neural_dtype="float32",
            scoring_dtype="float64",
            threads=4,
            interop_threads=1,
            deterministic=True,
            validated_hashes=pins,
        )
        raw, framed, targets, rows, provenance = source_scope(False)
        if provenance != fit["provenance"] or [len(rows[s]) for s in rows] != [4080, 398, 889]:
            raise ValueError("original full source/splits required")
        result["provenance"] = provenance
        catalogs, catalog_counts, indices, catalog_hashes = {}, {}, {}, {}
        for policy in POLICIES:
            catalogs[policy], catalog_counts[policy], indices[policy] = {}, {}, {}
            for split in rows:
                quartets, counts = build_quartets(rows[split], policy)
                catalogs[policy][split], catalog_counts[policy][split] = quartets, counts
                indices[policy][split] = query_indices(quartets)
                catalog_name = f"{policy}-{split}-catalog.npy"
                np.save(args.output / catalog_name, quartets)
                catalog_hashes[catalog_name] = sha(args.output / catalog_name)
        result["catalog_counts"] = catalog_counts
        result["catalog_hashes_frozen_before_scores"] = catalog_hashes
        cached, temperatures = {}, {}
        for arm in ["audio", "video_ordered"]:
            bounded(args.output, tick)
            model, values, val_logits, temp, checked = audit_arm(
                TRAINING / arm, arm, fit["arms"][arm], raw, framed, targets, rows
            )
            before = state_sha(model)
            cached[arm] = dict(
                validation=val_logits,
                train=verify_snapshot(
                    TRAINING / arm / "best-train.npz", rows["train"], targets["train"]
                ),
                calibration=verify_snapshot(
                    TRAINING / arm / "calibration.npz", rows["calibration"], targets["calibration"]
                ),
            )
            temperatures[arm] = temp
            if any(p.grad is not None for p in model.parameters()) or state_sha(model) != before:
                raise ValueError("parent recheck changed weights or gradients")
            result["models"][arm] = dict(
                checkpoint_sha256=fit["arms"][arm]["best_sha256"],
                best_epoch=fit["arms"][arm]["best_epoch"],
                temperatures=temp.tolist(),
                tensor_sha256=before,
                full_reload_checks=checked,
            )
            del model, values
        scores = {policy: {} for policy in POLICIES}
        for policy in POLICIES:
            for split in rows:
                query, videos, _ = indices[policy][split]
                scores[policy][split] = agreement_scores(
                    cached["audio"][split][query],
                    cached["video_ordered"][split][videos],
                    temperatures["audio"],
                    temperatures["video_ordered"],
                )
        cal_scores = np.concatenate([scores[p]["calibration"] for p in POLICIES])
        cal_targets = np.concatenate([indices[p]["calibration"][2] for p in POLICIES])
        temperature, calibration = select_query_temperature(cal_scores, cal_targets)
        if len(cal_scores) != 3184:
            raise ValueError("complete independent calibration required")
        result.update(query_temperature=temperature, calibration=calibration)
        for policy in POLICIES:
            report = dict(raw={}, calibrated={}, controls={})
            for split in rows:
                query, videos, y = indices[policy][split]
                report["raw"][split] = choice_metrics(
                    choice_probabilities(scores[policy][split]),
                    y,
                    choice_logps(scores[policy][split]),
                )
                report["calibrated"][split] = snapshot(
                    args.output,
                    policy + "-" + split,
                    scores[policy][split],
                    temperature,
                    query,
                    videos,
                    y,
                    rows[split],
                )["original_targets"]
            query, videos, y = indices[policy]["validation"]
            given = scores[policy]["validation"]
            matrix = given.reshape(-1, 4, 4)
            quartet = catalogs[policy]["validation"]
            controls = dict(
                audio_absent=(np.zeros_like(given), query, videos, None),
                video_absent=(np.zeros_like(given), query, videos, None),
                audio_swap=(
                    np.roll(matrix, -1, axis=1).reshape(-1, 4),
                    np.roll(quartet, -1, axis=1).ravel(),
                    videos,
                    (y + 1) % 4,
                ),
                video_swap=(
                    np.roll(matrix, -1, axis=2).reshape(-1, 4),
                    query,
                    np.roll(videos, -1, axis=1),
                    (y - 1) % 4,
                ),
            )
            for name, (value, q, v, corrected) in controls.items():
                bounded(args.output, tick)
                report["controls"][name] = snapshot(
                    args.output,
                    policy + "-" + name,
                    value,
                    temperature,
                    q,
                    v,
                    y,
                    rows["validation"],
                    corrected,
                )
                if corrected is not None:
                    current = report["controls"][name]["corrected_targets"]
                    matched = report["calibrated"]["validation"]
                    if any(
                        not np.isclose(current[k], matched[k], atol=1e-12, rtol=0)
                        for k in ["accuracy", "nll", "brier", "ece", "mean_confidence"]
                    ):
                        raise ValueError("corrected counterfactual metrics differ from matched")
            groups = [rows["validation"][i]["speaker_group"] for i in query]
            report["matched_minus_uniform"] = bootstrap_minus_uniform(
                choice_probabilities(given, temperature),
                y,
                groups,
                choice_logps(given, temperature),
            )
            order = [3, 0, 1, 2]
            for row in given:
                if choice_json(row, OPTION_IDS, temperature) != choice_json(
                    row[order], [OPTION_IDS[i] for i in order], temperature
                ):
                    raise ValueError("native custom-ID option permutation differs")
            report["json_option_permutation_exact"] = True
            (args.output / f"{policy}-example.json").write_text(
                choice_json(given[0], OPTION_IDS, temperature) + "\n", encoding="utf-8"
            )
            result["policies"][policy] = report
        if reference:
            for key in ["catalog_counts", "models", "query_temperature", "calibration", "policies"]:
                if result[key] != reference[key]:
                    raise ValueError("fresh correspondence reports differ")
            for name in reference["artifact_hashes"]:
                current_path, previous_path = args.output / name, args.replay_reference / name
                if name.endswith(".npz"):
                    with np.load(current_path, allow_pickle=False) as current:
                        with np.load(previous_path, allow_pickle=False) as prior:
                            if set(current.files) != set(prior.files) or any(
                                not np.array_equal(current[k], prior[k]) for k in current.files
                            ):
                                raise ValueError("fresh full prediction arrays differ")
                elif name.endswith(".npy"):
                    if not np.array_equal(
                        np.load(current_path, allow_pickle=False),
                        np.load(previous_path, allow_pickle=False),
                    ):
                        raise ValueError("fresh source catalog differs")
                elif sha(current_path) != sha(previous_path):
                    raise ValueError("fresh custom-ID JSON differs")
            recheck(
                {
                    str((args.replay_reference / k).resolve()): h
                    for k, h in reference["artifact_hashes"].items()
                }
            )
            if sha(args.replay_reference / "result.json") != args.replay_sha256:
                raise ValueError("primary correspondence changed")
        recheck({str(args.output / name): h for name, h in catalog_hashes.items()})
        recheck(pins)
        if source_scope(False)[4] != provenance or torch.cuda.is_initialized():
            raise ValueError("source authority changed or CUDA initialized")
        result.update(
            status="PASS_FROZEN_NATIVE_CORRESPONDENCE",
            fresh_process_replay=reference is not None,
            source_and_weights_unchanged=True,
            artifact_hashes={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        try:
            result.update(**bounded(args.output, tick))
        except Exception as error:
            result.update(status="FAIL_INCOMPLETE", resource_error=repr(error))
        result["ended_utc"] = datetime.now(UTC).isoformat()
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if result["status"] != "PASS_FROZEN_NATIVE_CORRESPONDENCE":
        raise RuntimeError("correspondence failed finalization")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
