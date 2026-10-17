"""Audit and compare completed fixed AV runs from saved development predictions.

No model loading, fitting, checkpoint selection changes or audit/test access.
"""

import argparse
import hashlib
import io
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.native_fit import summarize

ARMS = ("C0", "E0G0", "E0G1", "E1G0", "E1G1")
STEPS = (0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)
MODALITIES = ("text", "image", "audio", "video")
METRICS = ("accuracy", "nll", "brier", "ece")
COUNTS = {
    "train-joint": 23707,
    "train-common": 84480,
    "validation-joint": 8875,
    "validation-audio": 8875,
    "validation-video": 8875,
    "validation-common": 24346,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_receipts(receipts):
    require(set(receipts) == set(ARMS), "all five registered arms required")
    reference = receipts["C0"]
    shared_config = {k: v for k, v in reference["config"].items() if k != "arm"}
    for arm, r in receipts.items():
        require(
            r.get("arm") == arm
            and r.get("status") == "PASS_MATCHED_AV_TRAIN"
            and r.get("step") == r.get("optimizer_updates") == 32768
            and r.get("best_final_reload_verified") is True
            and r.get("encoder_updates") == r.get("cloud_cost") == 0
            and r.get("sealed_audit_loaded") is False
            and r.get("calibration_loaded") is False
            and r.get("error") is None,
            "complete untouched five-arm training and reload evidence required",
        )
        require(
            all(
                r.get(k) == reference.get(k) and r.get(k)
                for k in ("source_commit", "input_digest", "audit_sha256")
            ),
            "source/input/audit mismatch",
        )
        config = r["config"]
        require(
            config.get("arm") == arm
            and config.get("seed") == 17
            and config.get("max_steps") == 32768
            and {k: v for k, v in config.items() if k != "arm"} == shared_config,
            "fixed matched config required",
        )
        require(
            r.get("encoder_inclusive_parameters") == (196254874 if arm == "C0" else 197097949),
            "parameter budget mismatch",
        )


def finite_metrics(value):
    require(
        all(
            type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
            for k in METRICS
        ),
        "complete finite metrics required",
    )
    require(value["accuracy"] <= 1 and value["ece"] <= 1, "invalid probability metric")


def validate_curve(curve, recorded_best):
    require([p["step"] for p in curve] == list(STEPS), "all fixed evaluation points required")
    best_score, best_step = None, None
    initial = curve[0]["metrics"]["validation-common"]
    for p in curve:
        common = p["metrics"]["validation-common"]
        joint = p["metrics"]["validation-joint"]["macro"]
        finite_metrics(joint)
        finite_metrics(common["macro"])
        require(set(common["modality_macro"]) == set(MODALITIES), "four modalities required")
        for value in common["modality_macro"].values():
            finite_metrics(value)
        reasons = [
            m + "_accuracy_drop_over_2pp"
            for m in MODALITIES
            if common["modality_macro"][m]["accuracy"]
            < initial["modality_macro"][m]["accuracy"] - 0.02 - 1e-12
        ]
        if common["macro"]["nll"] > initial["macro"]["nll"] + 1e-5:
            reasons.append("existing_macro_nll_worse_than_initial")
        selected = not reasons and (best_score is None or joint["nll"] < best_score - 1e-5)
        expected = dict(
            eligible=not reasons, selected=selected, score=joint["nll"], reasons=reasons
        )
        require(p["selection"] == expected, "saved selector differs from fixed rule")
        if selected:
            best_score, best_step = joint["nll"], p["step"]
        require(p["best_step"] == best_step, "saved best step differs")
    require(best_step == recorded_best, "receipt best differs from curve")
    return best_step


def verify_artifacts(folder, receipt, required):
    artifacts = receipt.get("artifacts", {})
    require(required <= set(artifacts), "required artifact omitted from ledger")
    for name, expected in artifacts.items():
        require(
            name == Path(name).name and "/" not in name and "\\" not in name,
            "local artifact basenames required",
        )
        path = folder / name
        require(path.is_file() and sha(path) == expected, "artifact digest differs: " + name)


def bound_bytes(folder, receipt, name):
    payload = (folder / name).read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == receipt["artifacts"].get(name),
        "artifact changed during analysis: " + name,
    )
    return payload


def bound_json(folder, receipt, name):
    return json.loads(bound_bytes(folder, receipt, name))


def equal_summary(expected, actual):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), "metric keys differ")
        for key in expected:
            equal_summary(expected[key], actual[key])
    elif isinstance(expected, (int, float)):
        require(
            isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
            "recomputed metric differs",
        )
    else:
        require(expected == actual, "metric value differs")


def required_artifacts(arm, best):
    required = {
        "learning-curve.json",
        "evaluation-rows.json",
        "effective-config.json",
        "best.pt",
        "final.pt",
        "best.json",
        "best-reload.json",
        "final-reload.json",
    }
    required |= {f"checkpoint-{s:05}.pt" for s in STEPS}
    required |= {f"step-{s:05}-{label}.npy" for s in STEPS for label in COUNTS}
    for endpoint, step in (("best", best), ("final", 32768)):
        required |= {f"{endpoint}-step-{step:05}-{label}.npy" for label in COUNTS}
    if arm != "C0":
        labels = [label for label in COUNTS if "common" not in label]
        required |= {f"step-{s:05}-{label}-gates.npy" for s in STEPS for label in labels}
        for endpoint, step in (("best", best), ("final", 32768)):
            required |= {f"{endpoint}-step-{step:05}-{label}-gates.npy" for label in labels}
    return required


def verify_prediction(rows, payload, expected):
    scores = np.load(io.BytesIO(payload), allow_pickle=False)
    require(
        scores.dtype == np.float32 and scores.shape == (len(rows), 40),
        "fixed complete score matrix required",
    )
    equal_summary(expected, summarize(rows, scores))


def gate_summary(rows, gates):
    """Describe added-residual gates; -1 is bypass/padding, never a zero weight."""
    require(
        bool(rows)
        and gates.shape == (len(rows), 40, 2)
        and gates.dtype == np.float32
        and np.isfinite(gates).all(),
        "complete finite FP32 gate matrix required",
    )
    active = []
    for i, row in enumerate(rows):
        n = len(row["options"])
        require(
            2 <= n <= 40 and 0 <= row["target_index"] < n,
            "valid gate option/target binding required",
        )
        values = gates[i, :n]
        require(np.all(gates[i, n:] == -1), "gate padding sentinel differs")
        if np.all(values == -1):
            active.append(False)
        else:
            require(np.all((values >= 0) & (values <= 1)), "mixed bypass or invalid gate range")
            active.append(True)

    def stats(values):
        result = {}
        for k, name in enumerate(("audio_residual", "video_residual")):
            column = np.asarray(values, dtype=np.float64)[:, k]
            quantiles = np.quantile(column, [0, 0.05, 0.5, 0.95, 1], method="linear")
            result[name] = dict(
                mean=float(column.mean()),
                std=float(column.std(ddof=0)),
                **{
                    n: float(v)
                    for n, v in zip(("min", "q05", "median", "q95", "max"), quantiles, strict=True)
                },
            )
        return result

    def pool(indices):
        enabled = [i for i in indices if active[i]]
        result = dict(
            active_questions=len(enabled),
            inactive_questions=len(indices) - len(enabled),
            active_candidates=sum(len(rows[i]["options"]) for i in enabled),
        )
        names = (
            "candidate_weighted",
            "question_weighted",
            "target_candidates",
            "distractor_candidates",
        )
        if not enabled:
            return dict(result, **dict.fromkeys(names))
        values = [gates[i, : len(rows[i]["options"])] for i in enabled]
        result.update(
            candidate_weighted=stats(np.concatenate(values)),
            question_weighted=stats([v.mean(axis=0, dtype=np.float64) for v in values]),
            target_candidates=stats([gates[i, rows[i]["target_index"]] for i in enabled]),
            distractor_candidates=stats(
                np.concatenate(
                    [
                        np.delete(
                            gates[i, : len(rows[i]["options"])], rows[i]["target_index"], axis=0
                        )
                        for i in enabled
                    ]
                )
            ),
        )
        return result

    result = pool(list(range(len(rows))))
    tasks = sorted({r["source"] + ":" + r["task"] for r in rows})
    result["per_task"] = {
        task: pool([i for i, r in enumerate(rows) if r["source"] + ":" + r["task"] == task])
        for task in tasks
    }
    result["interpretation"] = (
        "Added-residual gates, not global modality importance or reliability."
    )
    return result


def load_curve(folder, receipt):
    required = required_artifacts(receipt["arm"], receipt["best_step"])
    verify_artifacts(folder, receipt, required)
    require(
        sha(folder / "effective-config.json") == receipt["config_sha256"],
        "effective config hash differs",
    )
    require(
        bound_json(folder, receipt, "effective-config.json") == receipt["config"],
        "effective config differs",
    )
    rows = bound_json(folder, receipt, "evaluation-rows.json")
    require(set(rows) == set(COUNTS), "complete evaluation views required")
    for label, count in COUNTS.items():
        require(
            len(rows[label]) == count and len({r["id"] for r in rows[label]}) == count,
            "fixed evaluation count or identity differs",
        )
        require(
            all(r["split"] == label.split("-")[0] for r in rows[label]), "evaluation split differs"
        )
    for label in ("validation-audio", "validation-video"):
        require(rows[label] == rows["validation-joint"], "ablation question order differs")
    train = {r["id"] for label in ("train-joint", "train-common") for r in rows[label]}
    validation = {
        r["id"] for label in ("validation-joint", "validation-common") for r in rows[label]
    }
    require(not train & validation, "train/validation question overlap")
    curve = bound_json(folder, receipt, "learning-curve.json")
    best = validate_curve(curve, receipt["best_step"])
    for p in curve:
        require(set(p["metrics"]) == set(COUNTS), "all recorded views required")
        diagnostics = {}
        for label in COUNTS:
            verify_prediction(
                rows[label],
                bound_bytes(folder, receipt, f"step-{p['step']:05}-{label}.npy"),
                p["metrics"][label],
            )
            if receipt["arm"] != "C0" and "common" not in label:
                gates = np.load(
                    io.BytesIO(
                        bound_bytes(folder, receipt, f"step-{p['step']:05}-{label}-gates.npy")
                    ),
                    allow_pickle=False,
                )
                diagnostics[label] = gate_summary(rows[label], gates)
                expected_active = (
                    0 if label in ("validation-audio", "validation-video") else len(rows[label])
                )
                require(
                    diagnostics[label]["active_questions"] == expected_active,
                    "joint/missing-stream gate bypass differs",
                )
                gated = receipt["arm"] in ("E0G1", "E1G1")
                diagnostics[label]["policy"] = "candidate_conditioned" if gated else "fixed_one"
                if not gated and expected_active:
                    require(
                        all(
                            v["min"] == v["max"] == 1
                            for v in diagnostics[label]["candidate_weighted"].values()
                        ),
                        "disabled gates must remain exactly one",
                    )
        p["gate_diagnostics"] = diagnostics
    best_record = bound_json(folder, receipt, "best.json")
    require(
        best_record["step"] == best
        and best_record["checkpoint_sha256"]
        == receipt["artifacts"][f"checkpoint-{best:05}.pt"]
        == receipt["artifacts"]["best.pt"],
        "selected checkpoint binding differs",
    )
    require(
        receipt["artifacts"]["final.pt"] == receipt["artifacts"]["checkpoint-32768.pt"],
        "final checkpoint binding differs",
    )
    for endpoint, step in (("best", best), ("final", 32768)):
        reload = bound_json(folder, receipt, endpoint + "-reload.json")
        require(reload["step"] == step, "reload step differs")
        original = next(p for p in curve if p["step"] == step)
        equal_summary(original["metrics"], reload["metrics"])
        for label in COUNTS:
            for suffix in ("", "-gates"):
                name = f"step-{step:05}-{label}{suffix}.npy"
                replay = endpoint + "-" + name
                if suffix and name not in receipt["artifacts"]:
                    continue
                x = np.load(io.BytesIO(bound_bytes(folder, receipt, name)), allow_pickle=False)
                y = np.load(io.BytesIO(bound_bytes(folder, receipt, replay)), allow_pickle=False)
                require(np.array_equal(x, y), "saved endpoint reload differs")
    return curve


def point_metrics(point):
    m = point["metrics"]
    return dict(
        joint_macro=m["validation-joint"]["macro"],
        common_macro=m["validation-common"]["macro"],
        common_modality_macro=m["validation-common"]["modality_macro"],
    )


def linear_combination(values, coefficients):
    first = next(iter(values.values()))
    if isinstance(first, dict):
        require(all(set(v) == set(first) for v in values.values()), "contrast metric keys differ")
        return {
            k: linear_combination({a: v[k] for a, v in values.items()}, coefficients) for k in first
        }
    return float(sum(coefficients[a] * value for a, value in values.items()))


def contrasts(points):
    pairs = (
        ("E1G1", "C0"),
        ("E0G0", "C0"),
        ("E1G0", "E0G0"),
        ("E1G1", "E0G1"),
        ("E0G1", "E0G0"),
        ("E1G1", "E1G0"),
    )
    values = {a: point_metrics(p) for a, p in points.items()}
    result = {
        f"{a}_minus_{b}": linear_combination({a: values[a], b: values[b]}, {a: 1, b: -1})
        for a, b in pairs
    }
    coefficients = dict(E1G1=1, E1G0=-1, E0G1=-1, E0G0=1)
    result["exchange_gate_interaction"] = linear_combination(
        {a: values[a] for a in coefficients}, coefficients
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="ARM=completed directory; all five"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    analysis_started = datetime.now(UTC).isoformat()
    source_files = [
        Path(__file__),
        repo / "scripts/native_fit.py",
        repo / "scripts/semantic_state_diagnostic.py",
    ]
    analysis_hashes = {str(p): sha(p) for p in source_files}
    require(not args.output.exists(), "new output required")
    pairs = [value.split("=", 1) for value in args.run]
    require(
        all(len(p) == 2 for p in pairs) and len(pairs) == len(ARMS), "five explicit runs required"
    )
    folders = {arm: Path(path) for arm, path in pairs}
    require(set(folders) == set(ARMS), "unique registered arms required")
    receipts = {
        arm: json.loads((folder / "result.json").read_text(encoding="utf-8"))
        for arm, folder in folders.items()
    }
    receipt_hashes = {arm: sha(folder / "result.json") for arm, folder in folders.items()}
    validate_receipts(receipts)
    curves = {arm: load_curve(folders[arm], receipts[arm]) for arm in ARMS}
    require(
        len({r["artifacts"]["evaluation-rows.json"] for r in receipts.values()}) == 1,
        "five-arm ordered evaluation metadata differs",
    )
    require(
        all(
            len({r["artifacts"][f"step-00000-{label}.npy"] for r in receipts.values()}) == 1
            for label in COUNTS
        ),
        "five-arm initial prediction matrices differ",
    )
    selected = {
        a: next(p for p in curves[a] if p["step"] == receipts[a]["best_step"]) for a in ARMS
    }
    result = dict(
        status="PASS_FIVE_ARM_SAVED_PREDICTION_COMPARISON",
        analysis_source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        analysis_source_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)
        ),
        analysis_source_sha256=analysis_hashes,
        analysis_started_utc=analysis_started,
        numpy_version=np.__version__,
        input_runs={a: str(folders[a]) for a in ARMS},
        source_commit=receipts["C0"]["source_commit"],
        input_digest=receipts["C0"]["input_digest"],
        receipt_sha256=receipt_hashes,
        configs={a: receipts[a]["config"] for a in ARMS},
        resources={
            a: {
                k: receipts[a].get(k)
                for k in (
                    "trainable_parameters",
                    "encoder_inclusive_parameters",
                    "config_sha256",
                    "device",
                    "versions",
                    "python",
                    "cuda",
                    "seconds",
                    "training_seconds",
                    "evaluation_seconds",
                    "peak_vram_bytes",
                    "peak_working_set_bytes",
                    "started_utc",
                    "finished_utc",
                    "resume_implemented",
                )
            }
            for a in ARMS
        },
        equal_step_contrasts={
            str(step): contrasts({a: curves[a][i] for a in ARMS}) for i, step in enumerate(STEPS)
        },
        selected_steps={a: p["step"] for a, p in selected.items()},
        selected_contrasts=contrasts(selected),
        curves=curves,
        inference_performed=False,
        optimizer_updates=0,
        sealed_audit_loaded=False,
        calibration_loaded=False,
        product_promoted=False,
        cloud_cost=0,
        model_reload_scope="Training-process best/final reload matrices rechecked; "
        "this analysis does not load a model in a fresh process.",
        caveat="Observed development validation; selected checkpoints can have unequal steps. "
        "Descriptive contrasts do not establish statistical certainty or both-required truth.",
    )
    require(
        all(sha(folders[a] / "result.json") == receipt_hashes[a] for a in ARMS),
        "receipt changed during analysis",
    )
    require(
        all(sha(p) == value for p, value in analysis_hashes.items()),
        "analysis source changed during execution",
    )
    result["analysis_finished_utc"] = datetime.now(UTC).isoformat()
    args.output.mkdir()
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(result["status"])


if __name__ == "__main__":
    main()
