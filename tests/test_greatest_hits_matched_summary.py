"""CPU-only contracts for the preregistered five-arm saved-result comparison."""

import copy
import hashlib
import io
import json
import sys

import numpy as np
import pytest

from scripts import summarize_greatest_hits_matched as summary
from scripts.native_fit import summarize
from scripts.summarize_greatest_hits_matched import (
    ARMS,
    STEPS,
    contrasts,
    required_artifacts,
    validate_curve,
    validate_receipts,
    verify_artifacts,
    verify_prediction,
)


def receipts():
    return {
        arm: {
            "arm": arm,
            "status": "PASS_MATCHED_AV_TRAIN",
            "step": 32768,
            "optimizer_updates": 32768,
            "encoder_updates": 0,
            "source_commit": "fixed-source",
            "input_digest": "fixed-input",
            "audit_sha256": "fixed-audit",
            "config": {"arm": arm, "seed": 17, "max_steps": 32768},
            "best_final_reload_verified": True,
            "sealed_audit_loaded": False,
            "calibration_loaded": False,
            "cloud_cost": 0,
            "encoder_inclusive_parameters": 196254874 if arm == "C0" else 197097949,
        }
        for arm in ARMS
    }


def metrics(nll=1.0, accuracy=0.8):
    value = dict(accuracy=accuracy, nll=nll, brier=0.2, ece=0.1)
    return dict(
        macro=dict(value),
        modality_macro={m: dict(value) for m in ("text", "image", "audio", "video")},
    )


def curve():
    points = []
    for step in STEPS:
        selected = step in (0, 512)
        joint = metrics(2.0 if step == 0 else 1.5 if step == 512 else 1.6)
        points.append(
            dict(
                step=step,
                metrics={"validation-joint": joint, "validation-common": metrics()},
                selection=dict(
                    eligible=True, selected=selected, score=joint["macro"]["nll"], reasons=[]
                ),
                best_step=0 if step == 0 else 512,
            )
        )
    point = points[2]
    point["metrics"]["validation-common"]["modality_macro"]["image"]["accuracy"] = 0.7
    point["metrics"]["validation-joint"]["macro"]["nll"] = 1.0
    point["selection"].update(
        eligible=False, selected=False, score=1.0, reasons=["image_accuracy_drop_over_2pp"]
    )
    return points


def test_five_complete_matched_runs_are_admitted():
    validate_receipts(receipts())


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "RUNNING"),
        ("step", 1024),
        ("optimizer_updates", 1024),
        ("best_final_reload_verified", False),
        ("encoder_updates", 1),
        ("sealed_audit_loaded", True),
        ("calibration_loaded", True),
        ("cloud_cost", 1),
        ("source_commit", "different"),
        ("input_digest", "different"),
        ("audit_sha256", "different"),
        ("encoder_inclusive_parameters", 200000001),
    ],
)
def test_incomplete_contaminated_or_unmatched_run_is_rejected(key, value):
    runs = receipts()
    runs["E1G1"][key] = value
    with pytest.raises(ValueError):
        validate_receipts(runs)


def test_missing_arm_or_changed_schedule_is_rejected():
    runs = receipts()
    del runs["E0G0"]
    with pytest.raises(ValueError):
        validate_receipts(runs)
    runs = receipts()
    runs["E0G1"]["config"]["seed"] = 23
    with pytest.raises(ValueError):
        validate_receipts(runs)


def test_selector_audit_keeps_early_eligible_best_and_rejects_better_ineligible():
    assert validate_curve(curve(), 512) == 512


@pytest.mark.parametrize("change", ["missing_point", "wrong_best", "wrong_reason", "adopt_bad"])
def test_curve_inconsistency_is_rejected(change):
    points = curve()
    if change == "missing_point":
        points.pop(3)
    elif change == "wrong_best":
        points[-1]["best_step"] = 32768
    elif change == "wrong_reason":
        points[2]["selection"]["reasons"] = []
    else:
        points[2]["selection"]["selected"] = True
    with pytest.raises(ValueError):
        validate_curve(points, 512)


def test_missing_modality_or_nonfinite_metric_cannot_satisfy_retention():
    points = curve()
    del points[1]["metrics"]["validation-common"]["modality_macro"]["audio"]
    with pytest.raises(ValueError):
        validate_curve(points, 512)
    points = curve()
    points[1]["metrics"]["validation-joint"]["macro"]["nll"] = float("nan")
    with pytest.raises(ValueError):
        validate_curve(points, 512)


def test_explicit_factorial_contrasts_have_candidate_minus_control_sign():
    points = {arm: copy.deepcopy(curve()[0]) for arm in ARMS}
    for arm, nll in zip(ARMS, (2.0, 1.9, 1.7, 1.5, 1.1), strict=True):
        points[arm]["metrics"]["validation-joint"]["macro"]["nll"] = nll
    result = contrasts(points)
    assert result["E1G1_minus_C0"]["joint_macro"]["nll"] == pytest.approx(-0.9)
    assert result["E0G0_minus_C0"]["joint_macro"]["nll"] == pytest.approx(-0.1)
    assert result["E1G1_minus_E0G1"]["joint_macro"]["nll"] == pytest.approx(-0.6)
    assert result["exchange_gate_interaction"]["joint_macro"]["nll"] == pytest.approx(-0.2)


def test_artifact_digest_change_or_missing_required_file_is_rejected(tmp_path):
    path = tmp_path / "scores.npy"
    path.write_bytes(b"saved predictions")
    receipt = {"artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    verify_artifacts(tmp_path, receipt, {path.name})
    path.write_bytes(b"different predictions")
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, receipt, {path.name})
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, receipt, {"missing.npy"})


def test_nonlocal_artifact_ledger_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, {"artifacts": {"../scores.npy": "x"}}, set())


def test_factorial_run_requires_gate_outputs_including_endpoint_reload():
    required = required_artifacts("E0G0", 512)
    assert "step-1024-validation-joint-gates.npy" not in required
    assert "step-01024-validation-joint-gates.npy" in required
    assert "best-step-00512-validation-joint-gates.npy" in required
    assert "final-step-32768-validation-video-gates.npy" in required
    assert not any("gates" in name for name in required_artifacts("C0", 512))


def test_saved_logits_are_recomputed_and_changed_predictions_or_padding_rejected():
    rows = [
        dict(
            id=str(i),
            source="source",
            task="task",
            modality="text",
            options=list(range(n)),
            target_index=t,
        )
        for i, (n, t) in enumerate(((2, 0), (3, 2)))
    ]
    scores = np.full((2, 40), -1e30, dtype=np.float32)
    scores[0, :2] = np.log([0.75, 0.25])
    scores[1, :3] = np.log([0.2, 0.3, 0.5])
    expected = summarize(rows, scores)
    assert expected["macro"]["nll"] == pytest.approx((-np.log(0.75) - np.log(0.5)) / 2)

    def payload(values):
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        return stream.getvalue()

    verify_prediction(rows, payload(scores), expected)
    changed = scores.copy()
    changed[0, 0] = 0
    with pytest.raises(ValueError):
        verify_prediction(rows, payload(changed), expected)
    changed = scores.copy()
    changed[0, 2] = -1
    with pytest.raises(ValueError):
        verify_prediction(rows, payload(changed), expected)
    with pytest.raises(ValueError):
        verify_prediction(rows, payload(scores.astype(np.float64)), expected)


def completed_fixture(tmp_path, monkeypatch, *, different_initial=False):
    """Exercise real CPU disk/CLI paths; tiny counts are only injected in this test."""
    monkeypatch.setattr(summary, "STEPS", (0, 32768))
    counts = {k: 4 if "common" in k else 2 for k in summary.COUNTS}
    monkeypatch.setattr(summary, "COUNTS", counts)
    runs = receipts()
    folders = {}
    for arm in ARMS:
        folder = tmp_path / arm
        folder.mkdir()
        folders[arm] = folder

        def write(name, data, folder=folder):
            (folder / name).write_text(json.dumps(data), encoding="utf-8")

        rows = {}
        for label, count in counts.items():
            split = label.split("-")[0]
            kind = "common" if "common" in label else "joint"
            rows[label] = [
                dict(
                    id=f"{split}-{kind}-{i}",
                    split=split,
                    options=["a", "b"],
                    target_index=0,
                    modality=("text", "image", "audio", "video")[i]
                    if kind == "common"
                    else "audio_video",
                    source=f"source-{i}",
                    task="task",
                )
                for i in range(count)
            ]
        write("evaluation-rows.json", rows)
        write("effective-config.json", runs[arm]["config"])
        points = []
        for step in summary.STEPS:
            metrics_by_view = {}
            for label, view in rows.items():
                scores = np.full((len(view), 40), -1e30, dtype=np.float32)
                probabilities = [0.8, 0.2] if different_initial and arm == "E1G1" else [0.75, 0.25]
                scores[:, :2] = np.log(probabilities)
                np.save(folder / f"step-{step:05}-{label}.npy", scores, allow_pickle=False)
                metrics_by_view[label] = summarize(view, scores)
                if arm != "C0" and "common" not in label:
                    gates = np.full((len(view), 40, 2), -1, dtype=np.float32)
                    gates[:, :2] = 0.5
                    np.save(folder / f"step-{step:05}-{label}-gates.npy", gates)
            points.append(
                dict(
                    step=step,
                    metrics=metrics_by_view,
                    best_step=0,
                    selection=dict(
                        eligible=True,
                        selected=step == 0,
                        score=metrics_by_view["validation-joint"]["macro"]["nll"],
                        reasons=[],
                    ),
                )
            )
            (folder / f"checkpoint-{step:05}.pt").write_bytes(str(step).encode())
        write("learning-curve.json", points)
        for endpoint, step in (("best", 0), ("final", 32768)):
            (folder / (endpoint + ".pt")).write_bytes(
                (folder / f"checkpoint-{step:05}.pt").read_bytes()
            )
            write(endpoint + "-reload.json", points[0 if step == 0 else 1])
            for path in folder.glob(f"step-{step:05}-*.npy"):
                (folder / (endpoint + "-" + path.name)).write_bytes(path.read_bytes())
        write("best.json", dict(step=0, checkpoint_sha256=summary.sha(folder / "best.pt")))
        runs[arm].update(
            best_step=0,
            config_sha256=summary.sha(folder / "effective-config.json"),
            artifacts={p.name: summary.sha(p) for p in folder.iterdir()},
        )
        write("result.json", runs[arm])
    return folders


def invoke_summary(monkeypatch, folders, output):
    args = ["summary", "--output", str(output)]
    for arm, path in folders.items():
        args += ["--run", f"{arm}={path}"]
    monkeypatch.setattr(sys, "argv", args)
    summary.main()


def test_complete_cpu_fixture_recomputes_and_compares_all_five_runs(tmp_path, monkeypatch):
    folders = completed_fixture(tmp_path, monkeypatch)
    output = tmp_path / "comparison"
    invoke_summary(monkeypatch, folders, output)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS_FIVE_ARM_SAVED_PREDICTION_COMPARISON"
    assert result["selected_steps"] == dict.fromkeys(ARMS, 0)
    assert set(result["equal_step_contrasts"]) == {"0", "32768"}
    assert result["selected_contrasts"]["E1G1_minus_C0"]["joint_macro"]["nll"] == 0
    assert not result["product_promoted"] and not result["inference_performed"]


def test_hash_consistent_but_unmatched_question_order_cannot_be_compared(tmp_path, monkeypatch):
    folders = completed_fixture(tmp_path, monkeypatch)
    folder = folders["E1G1"]
    path = folder / "evaluation-rows.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    for label in ("validation-joint", "validation-audio", "validation-video"):
        rows[label].reverse()
    path.write_text(json.dumps(rows), encoding="utf-8")
    receipt_path = folder / "result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][path.name] = summary.sha(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="ordered evaluation metadata differs"):
        invoke_summary(monkeypatch, folders, output)
    assert not output.exists()


def test_self_consistent_but_different_initial_outputs_cannot_be_compared(tmp_path, monkeypatch):
    folders = completed_fixture(tmp_path, monkeypatch, different_initial=True)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="initial prediction matrices differ"):
        invoke_summary(monkeypatch, folders, output)
    assert not output.exists()
