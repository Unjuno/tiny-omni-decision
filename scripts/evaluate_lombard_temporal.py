"""Fresh-process checkpoint replay, fixed controls and grouped development intervals."""

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

from scripts.lombard_readout_core import (
    OPTIONS,
    fit_normalizer,
    json_decision,
    select_temperatures,
    selected_epoch,
)
from scripts.lombard_readout_data import read_json, sha
from scripts.lombard_temporal_full import controls, json_read, verify_snapshot
from scripts.lombard_temporal_model import TemporalSlotHead
from scripts.report_lombard_grid_readouts import (
    _audit_metric,
    audit_optimizer_history,
    paired_bootstrap,
    source_permutation,
)
from scripts.robomnist_audio_cue import recheck
from scripts.train_lombard_temporal import (
    ARMS,
    bound_sources,
    guard,
    orders_for,
    predict,
    source_scope,
    state_sha,
    store_prediction,
    tensors,
)


def audit_arm(folder, arm, report, raw, framed, targets, rows):
    if (
        report["arm"] != arm
        or report["status"] != "PASS_ARM_EXECUTION"
        or report["updates"] != 768
        or report["consumed_assets"] != 97920
        or report["train_unique_assets"] != 4080
        or report["repeated_assets"] != 93840
        or report["head_parameters"] != 106355
        or not report["validation_evaluated"]
        or not report["calibration_evaluated"]
        or not report["final_reload_exact"]
        or not report["json_option_permutation_exact"]
    ):
        raise ValueError("complete fixed temporal arm required")
    recheck({str((folder / k).resolve()): v for k, v in report["artifact_hashes"].items()})
    orders = np.load(folder / "sample-orders.npy", allow_pickle=False)
    if not np.array_equal(orders, orders_for(4080, 24).numpy()):
        raise ValueError("actual ordered sampler differs")
    entries = [
        json.loads(line) for line in (folder / "optimizer-history.jsonl").read_text().splitlines()
    ]
    audit_optimizer_history(entries)
    history = report["history"]
    if [r["epoch"] for r in history] != list(range(1, 25)):
        raise ValueError("all 24 observations required")
    for entry in history:
        if entry["consumed_assets"] != entry["epoch"] * 4080:
            raise ValueError("epoch consumption differs")
        _audit_metric(entry["train_fixed"], 4080)
        _audit_metric(entry["validation"], 889)
        verify_snapshot(
            folder / f"epoch-{entry['epoch']:02}.npz",
            rows["validation"],
            targets["validation"],
            entry["validation"],
        )
    best = selected_epoch(history)
    if (
        report["best_epoch"] != best
        or report["selected_consumed_assets"] != best * 4080
        or report["selected_consumed_slot_decisions"] != best * 4080 * 6
    ):
        raise ValueError("selected checkpoint/counts differ")
    normalizer = fit_normalizer(*raw["train"])
    with np.load(folder / "normalizer.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(normalizer) or any(
            not np.array_equal(saved[k], normalizer[k]) for k in normalizer
        ):
            raise ValueError("normalization differs from original training-only fit")
    values = tensors(raw, framed, normalizer, arm)
    model = TemporalSlotHead().eval()
    for name, epoch in [("final", 24), ("best", best)]:
        state = torch.load(folder / f"{name}.pt", weights_only=True, map_location="cpu")
        if sha(folder / f"{name}.pt") != report[f"{name}_sha256"]:
            raise ValueError("checkpoint sha differs")
        model.load_state_dict(state, strict=True)
        before = state_sha(model)
        optimizer = torch.load(
            folder / f"{name}-optimizer.pt", weights_only=True, map_location="cpu"
        )
        if optimizer["epoch"] != epoch or optimizer["global_updates"] != epoch * 32:
            raise ValueError("checkpoint/optimizer step binding differs")
        if any(float(s["step"]) != epoch * 32 for s in optimizer["optimizer"]["state"].values()):
            raise ValueError("optimizer parameter step differs")
        expected_train = verify_snapshot(
            folder / f"{name}-train.npz", rows["train"], targets["train"]
        )
        if not np.array_equal(predict(model, values["train"]), expected_train):
            raise ValueError("fresh-process train logits differ")
        expected_val = verify_snapshot(
            folder / f"{name}-raw.npz",
            rows["validation"],
            targets["validation"],
            report[f"{name}_raw"],
        )
        if not np.array_equal(predict(model, values["validation"]), expected_val):
            raise ValueError("fresh-process validation logits differ")
        if state_sha(model) != before:
            raise ValueError("inference changed model tensors")
    cal = predict(model, values["calibration"])
    if not np.array_equal(
        cal,
        verify_snapshot(folder / "calibration.npz", rows["calibration"], targets["calibration"]),
    ):
        raise ValueError("fresh-process calibration logits differ")
    temps = select_temperatures(cal, targets["calibration"])
    if not np.array_equal(temps, report["temperatures"]):
        raise ValueError("selected calibration temperatures differ")
    logits = verify_snapshot(
        folder / "best-calibrated.npz",
        rows["validation"],
        targets["validation"],
        report["best_calibrated"],
        temps,
        replayed_logits=expected_val,
    )
    option_order = {s: list(reversed(o)) for s, o in OPTIONS.items()}
    for value in logits:
        if json.loads(json_decision(value, temperatures=temps)) != json.loads(
            json_decision(value, option_order, temps)
        ):
            raise ValueError("full-cohort native option permutation differs")
    return (
        model,
        values,
        logits,
        temps,
        dict(
            all_saved_epoch_metrics_exact=True,
            best_final_train_validation_bitwise_exact=True,
            calibration_logits_and_temperatures_exact=True,
            optimizer_and_sampler_accounting_exact=True,
            full_validation_json_permutation_exact=True,
            original_requires_grad_retained=True,
            train_validation_macro_accuracy_gap=(
                history[best - 1]["train_fixed"]["macro"]["accuracy"]
                - report["best_raw"]["macro"]["accuracy"]
            ),
            train_validation_macro_nll_gap=(
                report["best_raw"]["macro"]["nll"]
                - history[best - 1]["train_fixed"]["macro"]["nll"]
            ),
            best_epoch=best,
            selected_consumed_assets=best * 4080,
            total_consumed_assets=97920,
            selected_raw=report["best_raw"],
            selected_calibrated=report["best_calibrated"],
            final_raw=report["final_raw"],
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(
        status="RUNNING",
        device="cpu",
        gpu_used=False,
        training_updates=0,
        encoders_loaded=False,
        sealed_audit_opened=False,
        arms={},
        controls={},
        started_utc=datetime.now(UTC).isoformat(),
        command=subprocess.list2cmdline(
            [sys.executable, "-m", "scripts.evaluate_lombard_temporal", *sys.argv[1:]]
        ),
        training_sha256=args.training_sha256,
        training_path=str(args.training.resolve()),
        scope="observed development validation; selected single-seed models; not blind",
    )
    try:
        training = read_json(args.training / "result.json", args.training_sha256)
        if training["status"] != "PASS_FIVE_ARM_TRAINING" or list(training["arms"]) != ARMS:
            raise ValueError("complete admitted five-arm fit required")
        pins = training["validated_hashes"]
        recheck(pins)
        recheck(training["full_execution_hashes"])
        if bound_sources() != pins:
            raise ValueError("fresh process implementations differ")
        if sha(args.training / "effective-config.json") != training["effective_config_sha256"]:
            raise ValueError("training effective config differs")
        result.update(
            validated_hashes=pins, full_execution_hashes=training["full_execution_hashes"]
        )
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        raw, framed, targets, rows, provenance = source_scope(False)
        if provenance != training["provenance"]:
            raise ValueError("full source/cache provenance differs")
        permutation = source_permutation(rows["validation"])
        result["donor_source_ids"] = [rows["validation"][i]["source_id"] for i in permutation]
        selected = {}
        artifacts = {
            str((args.training / "effective-config.json").resolve()): training[
                "effective_config_sha256"
            ],
            str((args.training / "result.json").resolve()): args.training_sha256,
        }
        for arm in ARMS:
            guard(args.output, tick)
            folder = args.training / arm
            arm_report = json_read(folder / "result.json")
            if arm_report != training["arms"][arm]:
                raise ValueError("arm/controller reports differ")
            artifacts[str((folder / "result.json").resolve())] = sha(folder / "result.json")
            artifacts.update(
                {str((folder / k).resolve()): v for k, v in arm_report["artifact_hashes"].items()}
            )
            model, values, logits, temps, checked = audit_arm(
                folder, arm, arm_report, raw, framed, targets, rows
            )
            result["arms"][arm] = checked
            selected[arm] = (logits, temps)
            if arm.startswith("joint"):
                before = state_sha(model)
                result["controls"][arm] = {}
                for name, value in controls(values["validation"], permutation).items():
                    guard(args.output, tick)
                    actual = predict(model, value)
                    if name == "matched" and not np.array_equal(actual, logits):
                        raise ValueError("matched-control prediction differs")
                    result["controls"][arm][name] = store_prediction(
                        args.output,
                        arm + "-" + name,
                        actual,
                        targets["validation"],
                        rows["validation"],
                        temps,
                    )
                if state_sha(model) != before:
                    raise ValueError("fixed controls changed selected weights")
            del model, values
            print("verified", arm, flush=True)
        result["intervals"] = {}
        for first, second in [
            ("video_broadcast", "video_ordered"),
            ("joint_broadcast", "joint_ordered"),
            ("audio", "joint_broadcast"),
            ("audio", "joint_ordered"),
        ]:
            first_logits, first_temp = selected[first]
            second_logits, second_temp = selected[second]
            result["intervals"][second + "_minus_" + first] = paired_bootstrap(
                first_logits,
                second_logits,
                rows["validation"],
                first_temp,
                second_temp,
                replicates=10000,
                seed=17,
            )
        recheck(pins)
        recheck(training["full_execution_hashes"])
        recheck(artifacts)
        if (
            sha(args.training / "result.json") != args.training_sha256
            or source_scope(False)[4] != provenance
            or torch.cuda.is_initialized()
        ):
            raise ValueError("training/source authority changed or CUDA initialized")
        result.update(
            status="PASS_FRESH_REPLAY_CONTROLS_AND_GROUPED_COMPARISON",
            source_and_training_unchanged=True,
            model_tensors_unchanged=True,
            artifact_hashes={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        try:
            result.update(**guard(args.output, tick))
        except Exception as error:
            result.update(status="FAIL_INCOMPLETE", resource_error=repr(error))
        result["ended_utc"] = datetime.now(UTC).isoformat()
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if result["status"] != "PASS_FRESH_REPLAY_CONTROLS_AND_GROUPED_COMPARISON":
        raise RuntimeError("comparison failed resource finalization")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
