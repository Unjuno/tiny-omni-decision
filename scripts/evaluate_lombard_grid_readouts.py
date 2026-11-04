"""Audit all three completed arms, then evaluate frozen selected-model controls once."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scripts.lombard_readout_core import (
    SlotHead,
    fit_normalizer,
    make_inputs,
    select_temperatures,
    summarize,
    targets_from_rows,
)
from scripts.lombard_readout_data import load_data, read_json, sha
from scripts.report_lombard_grid_readouts import (
    audit_optimizer_history,
    audit_summary,
    paired_bootstrap,
    source_permutation,
    validate_saved_predictions,
)
from scripts.train_lombard_grid_readouts import predict, sample_orders, save_predictions, state_sha

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def report_identity(provenance):
    if provenance.get("test_fixture_only"):
        return (
            "PASS_MECHANICAL_FIXTURE_REPORT",
            "artificial test inputs only; not model quality or native corpus evidence",
        )
    return (
        "PASS_FIXED_MODEL_REPORT",
        "native spoken-command slots; pooled features, reused development validation "
        "and selected/calibrated weights, not blind/generic Omni",
    )


def check_artifact_hashes(folder, report):
    for name, key in [
        ("best.pt", "best_sha256"),
        ("final.pt", "final_sha256"),
        ("normalization.npz", "normalizer_sha256"),
    ]:
        if sha(folder / name) != report[key]:
            raise ValueError(f"checkpoint/normalizer hash differs: {name}")


def audit_prediction_bundle(folder, prefix, replay, rows, temperatures=None):
    stored = np.load(folder / (prefix + "-logits.npy"), allow_pickle=False)
    if stored.dtype != replay.dtype or not np.array_equal(stored, replay):
        raise ValueError(f"saved logits differ from checkpoint replay: {prefix}")
    probabilities = summarize(replay, targets_from_rows(rows), temperatures)[1]
    saved = np.load(folder / (prefix + "-probabilities.npy"), allow_pickle=False)
    if saved.dtype != probabilities.dtype or not np.array_equal(saved, probabilities):
        raise ValueError(f"saved probabilities differ from checkpoint replay: {prefix}")
    return validate_saved_predictions(
        replay, read_json(folder / (prefix + "-predictions.json")), rows, temperatures,
    )


def joint_controls(model, audio, video, normalizer, rows, temperatures, output, *, device="cuda"):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("selected inference model must be eval and frozen")
    before = state_sha(model)
    output.mkdir(exist_ok=False)
    permutation = source_permutation(rows)
    (output / "source-permutation.json").write_text(json.dumps(permutation), encoding="utf-8")
    conditions = {
        "matched": make_inputs(audio, video, "joint", normalizer),
        "audio_mismatch": make_inputs(audio[permutation], video, "joint", normalizer),
        "video_mismatch": make_inputs(audio, video[permutation], "joint", normalizer),
        "audio_missing": make_inputs(audio, video, "video", normalizer),
        "video_missing": make_inputs(audio, video, "audio", normalizer),
    }
    results = {}
    for name, value in conditions.items():
        logits = predict(model, torch.from_numpy(value).to(device))
        metric = save_predictions(output, name, logits, targets_from_rows(rows), rows, temperatures)
        if audit_prediction_bundle(output, name, logits, rows, temperatures) != metric:
            raise ValueError("saved control metrics differ")
        results[name] = dict(
            metrics=metric,
            logits_sha256=sha(output / (name + "-logits.npy")),
            predictions_sha256=sha(output / (name + "-predictions.json")),
            probabilities_sha256=sha(output / (name + "-probabilities.npy")),
        )
    if state_sha(model) != before:
        raise ValueError("selected inference weights changed during controls")
    report = dict(
        conditions=results,
        model_unchanged=True,
        model_tensor_sha256=before,
        temperatures=np.asarray(temperatures).tolist(),
        source_permutation_sha256=sha(output / "source-permutation.json"),
        missing_stream_scope="unseen masks; diagnostic OOD, not trained missing-input support",
        mismatch_scope="p/l-preserving cross-group donors; paired dependencies remain",
    )
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    run = ROOT / "lombard-grid-pooled-readouts-01"
    controller = read_json(run / "result.json")
    if (
        controller.get("status") != "PASS_THREE_ARM_EXECUTION"
        or controller.get("smoke") is not False
    ):
        raise ValueError("completed three-arm full learning required before any report evaluation")
    raw, targets, rows, provenance = load_data(ROOT)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise ValueError("idle local CUDA required for same-backend fixed-model replay")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = ROOT / "lombard-grid-pooled-readouts-report-01"
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(
        status="RUNNING",
        arms={},
        comparisons={},
        source_result_sha256=sha(run / "result.json"),
        provenance=provenance,
        script_sha256=sha(Path(__file__)),
        training=False,
        cloud_cost=0,
    )
    logits, temps, models, normalizers = {}, {}, {}, {}
    try:
        for name in ["audio", "video", "joint"]:
            folder = run / name
            report = read_json(folder / "result.json")
            if report != controller["arms"][name]:
                raise ValueError("arm report differs from completed controller")
            audit_summary(report, provenance)
            check_artifact_hashes(folder, report)
            optimizer_log = [
                json.loads(line)
                for line in (folder / "optimizer-history.jsonl").read_text().splitlines()
            ]
            audit_optimizer_history(optimizer_log)
            orders = np.load(folder / "sampling-orders.npy", allow_pickle=False)
            if sha(folder / "sampling-orders.npy") != report[
                "sampling_order_sha256"
            ] or not np.array_equal(orders, sample_orders(4080, 24).numpy()):
                raise ValueError("actual seed17 sampling order differs")
            expected_order = [r["source_id"] for r in rows["train"]]
            if read_json(folder / "train-ids.json") != expected_order:
                raise ValueError("training source order differs")
            with np.load(folder / "normalization.npz", allow_pickle=False) as archive:
                norm = {k: archive[k] for k in archive.files}
            expected_norm = fit_normalizer(*raw["train"])
            if set(norm) != set(expected_norm) or any(
                not np.array_equal(norm[k], expected_norm[k]) for k in norm
            ):
                raise ValueError("normalization differs from full train-only fit")
            model = SlotHead().to("cuda")
            model.load_state_dict(
                torch.load(folder / "best.pt", map_location="cuda", weights_only=True), strict=True
            )
            model.eval().requires_grad_(False)
            if state_sha(model) != report["selected_tensor_sha256"]:
                raise ValueError("reloaded selected tensor hash differs")
            cal = predict(
                model, torch.from_numpy(make_inputs(*raw["calibration"], name, norm)).cuda()
            )
            if not np.array_equal(
                cal, np.load(folder / "calibration-logits.npy", allow_pickle=False)
            ):
                raise ValueError("selected calibration checkpoint replay differs")
            temperatures = select_temperatures(cal, targets["calibration"])
            if not np.array_equal(temperatures, report["temperatures"]):
                raise ValueError("calibration rule/result differs")
            value = predict(
                model, torch.from_numpy(make_inputs(*raw["validation"], name, norm)).cuda()
            )
            final_model = SlotHead().to("cuda")
            final_model.load_state_dict(
                torch.load(folder / "final.pt", map_location="cuda", weights_only=True), strict=True
            )
            final_model.eval().requires_grad_(False)
            final_logits = predict(
                final_model, torch.from_numpy(make_inputs(*raw["validation"], name, norm)).cuda()
            )
            for prefix, replay, t, key in [
                ("best-raw", value, None, "best_raw"),
                ("best-calibrated", value, temperatures, "best_calibrated"),
                ("final-raw", final_logits, None, "final_raw"),
                ("final-same-temperature", final_logits, temperatures, "final_same_temperature"),
            ]:
                computed = audit_prediction_bundle(
                    folder, prefix, replay, rows["validation"], t,
                )
                if computed != report[key]:
                    raise ValueError(f"saved prediction metrics differ: {prefix}")
            metric = report["best_calibrated"]
            epoch_metric = report["history"][report["best_epoch"] - 1]["validation"]
            if (
                report["best_raw"] != epoch_metric
                or report["final_raw"] != report["history"][-1]["validation"]
            ):
                raise ValueError("best/final checkpoint metrics differ from fixed learning curve")
            subgroups = {}
            for key in ["condition", "speaker", "speaker_group"]:
                subgroups[key] = {}
                for group in sorted({r[key] for r in rows["validation"]}):
                    idx = [i for i, r in enumerate(rows["validation"]) if r[key] == group]
                    subgroups[key][group] = summarize(
                        value[idx], targets["validation"][idx], temperatures
                    )[0]
            logits[name], temps[name], models[name], normalizers[name] = (
                value,
                temperatures,
                model,
                norm,
            )
            result["arms"][name] = dict(
                best=metric,
                final=report["final_same_temperature"],
                best_epoch=report["best_epoch"],
                accounting=report["accounting"],
                validation_subgroups=subgroups,
                checkpoint_replay_bitwise=True,
            )
        if (
            len({controller["arms"][k]["initial_tensor_sha256"] for k in logits}) != 1
            or len({controller["arms"][k]["sampling_order_sha256"] for k in logits}) != 1
        ):
            raise ValueError("arm initialization/sampling differ")
        for baseline in ["audio", "video"]:
            result["comparisons"]["joint_minus_" + baseline] = paired_bootstrap(
                logits[baseline],
                logits["joint"],
                rows["validation"],
                temps[baseline],
                temps["joint"],
            )
        controls_dir = out / "joint-controls"
        controls = joint_controls(
            models["joint"],
            *raw["validation"],
            normalizers["joint"],
            rows["validation"],
            temps["joint"],
            controls_dir,
        )
        if not np.array_equal(
            logits["joint"], np.load(controls_dir / "matched-logits.npy", allow_pickle=False)
        ):
            raise ValueError("control baseline differs from selected fixed-model replay")
        result["joint_controls"] = controls
        for name in ["audio_mismatch", "video_mismatch", "audio_missing", "video_missing"]:
            changed = np.load(controls_dir / (name + "-logits.npy"), allow_pickle=False)
            result["comparisons"][name + "_minus_matched"] = paired_bootstrap(
                logits["joint"], changed, rows["validation"], temps["joint"], temps["joint"]
            )
        after = load_data(ROOT)[3]
        if after != provenance:
            raise ValueError("source/feature hashes changed during report evaluation")
        status, scope = report_identity(provenance)
        result.update(
            status=status,
            all_source_hashes_unchanged=True,
            test_fixture_only=bool(provenance.get("test_fixture_only")),
            quality_scope=scope,
        )
    except Exception as error:
        result.update(status="FAIL_INCOMPLETE", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.perf_counter() - started
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["arms", "comparisons", "provenance"]}
        )
    )


if __name__ == "__main__":
    main()
