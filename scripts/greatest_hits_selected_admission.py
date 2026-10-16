"""CPU-only selected-run and natural-reference binding, without model selection."""

import numpy as np

from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS

ARMS = ("C0", "E0G0", "E0G1", "E1G0", "E1G1")


def require_complete(run):
    if (
        run.get("status") != "PASS_MATCHED_AV_TRAIN"
        or run.get("arm") not in ARMS
        or run.get("step") != MAX_STEPS
        or run.get("optimizer_updates") != MAX_STEPS
        or run.get("encoder_updates") != 0
        or run.get("best_final_reload_verified") is not True
        or type(run.get("best_step")) is not int
        or run["best_step"] not in EVALUATION_STEPS
    ):
        raise ValueError("completed full matched training and verified selection required")


def require_selection_set(runs):
    if set(runs) != set(ARMS):
        raise ValueError("all five frozen selections required before diagnostics")
    for arm, run in runs.items():
        require_complete(run)
        if run["arm"] != arm or run["input_digest"] != runs["C0"]["input_digest"]:
            raise ValueError("matched arm and identical input digest required")
        for field in ("source_commit", "versions"):
            if not run.get(field) or run[field] != runs["C0"].get(field):
                raise ValueError("matched training source and runtime required")
        if not isinstance(run.get("config"), dict) or {
            k: v for k, v in run["config"].items() if k != "arm"
        } != {k: v for k, v in runs["C0"]["config"].items() if k != "arm"}:
            raise ValueError("matched training conditions required")


def require_selected(run, best, checkpoint, checkpoint_sha256):
    require_complete(run)
    step = run["best_step"]
    if (
        best["step"] != step
        or checkpoint["step"] != step
        or best["checkpoint"] != f"checkpoint-{step:05}.pt"
        or best["checkpoint_sha256"] != checkpoint_sha256
        or checkpoint["sampler_sha256"] != run["config"]["plan_sha256"]
        or checkpoint["config_sha256"] != run["config_sha256"]
    ):
        raise ValueError("selected checkpoint/config/sampler binding differs")


def natural_reference(rows, saved_rows, scores):
    ids = {r["id"] for r in rows}
    if (
        not rows
        or len(ids) != len(rows)
        or len({r["id"] for r in saved_rows}) != len(saved_rows)
        or {r["id"] for r in saved_rows} != ids
        or scores.shape != (len(rows), 40)
        or scores.dtype != np.float32
        or not np.isfinite(scores).all()
    ):
        raise ValueError("complete unique natural reference identities/FP32 scores required")
    by_id = {r["id"]: r for r in rows}
    result = {}
    for row, values in zip(saved_rows, scores, strict=True):
        expected = by_id[row["id"]]
        if (
            row["split"] != "validation"
            or expected["split"] != "validation"
            or row["options"] != expected["options"]
            or not 2 <= len(row["options"]) <= 40
            or not np.all(values[len(row["options"]) :] == np.float32(-1e30))
        ):
            raise ValueError("natural reference split/options/padding differs")
        result[row["id"]] = values[: len(row["options"])].copy()
    return result
