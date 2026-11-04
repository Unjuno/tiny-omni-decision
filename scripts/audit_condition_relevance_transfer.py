"""Independent CPU audit of terminal fixed-transfer predictions and metrics."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from scripts.condition_relevance_inference import evaluation_groups
from scripts.condition_relevance_postfit_io import (
    _json,
    _no_redirect,
    _outside_protected,
    _require,
    _simple_name,
    _verified_bytes,
    _verify_stream,
    capture_execution_source,
    read_frozen_transfer,
    verify_pair_files,
)


def sha(path):
    return hashlib.sha256(_no_redirect(path).read_bytes()).hexdigest()


def independent_metrics(scores, target):
    """Scalar math implementation; no call to the evaluator's metric routine."""
    values = np.asarray(scores, dtype=np.float64)
    _require(
        values.ndim == 2 and len(values) > 0 and values.shape[1] >= 2
        and len(target) == len(values) and np.isfinite(values).all(),
        "complete finite score matrix required",
    )
    _require(
        all(type(t) is int and 0 <= t < values.shape[1] for t in target),
        "integer target indices required",
    )
    probabilities, correct, confidence, losses, briers = [], [], [], [], []
    bins = [[] for _ in range(15)]
    for index, (row, truth) in enumerate(zip(values.tolist(), target, strict=True)):
        maximum = max(row)
        exp_values = [math.exp(v - maximum) for v in row]
        denominator = math.fsum(exp_values)
        probability = [v / denominator for v in exp_values]
        prediction = row.index(maximum)
        probabilities.append(probability)
        correct.append(float(prediction == truth))
        confidence.append(probability[prediction])
        losses.append(maximum - row[truth] + math.log(denominator))
        briers.append(math.fsum((p - (i == truth)) ** 2 for i, p in enumerate(probability)))
        bins[min(int(confidence[-1] * 15), 14)].append(index)
    count = len(values)
    ece = math.fsum(
        abs(math.fsum(correct[i] - confidence[i] for i in members)) / count
        for members in bins if members
    )
    return dict(
        count=count, accuracy=math.fsum(correct) / count, nll=math.fsum(losses) / count,
        brier=math.fsum(briers) / count, ece=ece,
        mean_confidence=math.fsum(confidence) / count,
    ), probabilities


def _compare(expected, actual):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(expected) == set(actual), "metric keys differ")
        return max((_compare(v, actual[k]) for k, v in expected.items()), default=0)
    if type(expected) in (int, float):
        _require(type(actual) in (int, float) and math.isfinite(actual), "finite metric required")
        if type(expected) is int:
            _require(type(actual) is int and actual == expected, "metric count differs")
        error = abs(expected - actual)
        _require(error <= 1e-10, "saved metric differs from independent calculation")
        return error
    _require(actual == expected, "metric scope differs")
    return 0


def verify_saved_evidence(rows, scores, predictions, summary):
    scores = np.asarray(scores)
    _require(
        bool(rows) and scores.shape == (len(rows), 40) and len(predictions) == len(rows),
        "complete aligned rows, predictions and scores required",
    )
    target = [r["options"].index(r["target"]) for r in rows]
    overall, probabilities = independent_metrics(scores, target)
    maximum_probability_error = 0.0
    for row, values, saved, probability in zip(
        rows, scores, predictions, probabilities, strict=True
    ):
        _require(isinstance(saved, dict), "saved prediction object required")
        for key in ("id", "group", "condition", "field", "target", "target_id", "option_ids"):
            _require(saved.get(key) == row[key], "saved prediction identity differs: " + key)
        prediction = int(values.argmax())
        _require(
            saved.get("prediction") == row["options"][prediction]
            and saved.get("predicted_id") == row["option_ids"][prediction],
            "saved predicted label differs",
        )
        saved_p = np.asarray(saved.get("probabilities"), dtype=np.float64)
        _require(saved_p.shape == (40,) and np.isfinite(saved_p).all(), "finite40 probabilities")
        error = float(np.max(np.abs(saved_p - probability)))
        _require(error <= 1e-12, "saved probabilities differ")
        maximum_probability_error = max(maximum_probability_error, error)

    def grouped(key):
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(key(row), []).append(i)
        return {
            name: independent_metrics(scores[indices], [target[i] for i in indices])[0]
            for name, indices in groups.items()
        }

    fields = grouped(lambda r: r["condition"] + ":" + r["field"])
    expected = dict(
        overall=overall,
        conditions=grouped(lambda r: r["condition"]),
        condition_fields=fields,
        actual_query_slices=grouped(
            lambda r: f"{r['condition']}:{r['field']}:q{r['query_index']}:{r['option_form']}"
        ),
        macro_condition_field={
            key: math.fsum(m[key] for m in fields.values()) / len(fields)
            for key in ("accuracy", "nll", "brier", "ece")
        },
        minimum_condition_field_accuracy=min(m["accuracy"] for m in fields.values()),
        scope="authored development transfer; no calibration or checkpoint reselection",
    )
    return dict(
        rows=len(rows), condition_fields=len(fields),
        maximum_metric_error=_compare(expected, summary),
        maximum_probability_error=maximum_probability_error,
    )


def inspect_transfer(directory, result_sha, comparison, comparison_sha):
    directory = _no_redirect(directory)
    result = _json(directory / "result.json", result_sha)
    _require(
        result.get("status") == "PASS_FIXED_CONDITION_RELEVANCE_TRANSFER"
        and result.get("mode") == "transfer",
        "completed fixed transfer required",
    )
    for key in ("sealed_audit_loaded", "calibration_loaded", "calibration_applied",
                "checkpoint_reselected", "product_complete", "blind_evaluation"):
        _require(result.get(key) is False, "transfer invariant changed: " + key)
    _require(
        result.get("optimizer_updates") == 0 and result.get("cloud_cost") == 0
        and result.get("transfer_loaded") is True,
        "inference-only transfer required",
    )
    comparison = _no_redirect(comparison)
    receipt = _json(comparison / "result.json", comparison_sha)
    _require(
        receipt.get("status") == "PASS_FROZEN_CONDITION_RELEVANCE_COMPARISON"
        and result.get("comparison_sha256") == comparison_sha,
        "original frozen comparison required",
    )
    execution = capture_execution_source(
        receipt["source_root"], __file__, "audit_condition_relevance_transfer"
    )
    for path, expected in receipt["code_pins"].items():
        _verify_stream(path, expected)
    checked = verify_pair_files(
        receipt["run_directories"], receipt["result_hashes"], receipt["source_root"]
    )
    freeze_sha = receipt["artifacts"]["comparison-freeze.json"]
    frozen = _json(comparison / "comparison-freeze.json", freeze_sha)
    _require(frozen == checked["frozen"], "frozen identities differ")
    groups = evaluation_groups(frozen["roles"])
    _require(
        result.get("frozen_sha256") == freeze_sha
        and result.get("all_roles") == frozen["roles"]
        and result.get("rows") == 16384
        and result.get("unique_weights_evaluated") == len(groups)
        and result.get("pair_forwards") == 16384 * 40 * len(groups),
        "complete frozen roles and forwards required",
    )
    expected_files = {"transfer-accounting.json", "token-preflight.json"} | {
        g["canonical_role"] + "-" + suffix for g in groups
        for suffix in ("scores.npy", "predictions.jsonl", "metrics.json")
    }
    _require(set(result["artifacts"]) == expected_files, "complete output manifest required")
    for name, expected in result["artifacts"].items():
        _require(_simple_name(name), "simple output filename required")
        _verify_stream(directory / name, expected)
    raw = read_frozen_transfer(
        comparison / "comparison-freeze.json", freeze_sha,
        receipt["run_directories"], receipt["result_hashes"], receipt["source_root"],
    )
    rows = [json.loads(line) for line in raw.splitlines()]
    _require(len(rows) == 16384 and len({r["group"] for r in rows}) == 256, "complete state count")
    _require(set(result["metrics_by_role"]) == set(frozen["roles"]), "all role metrics required")
    audits = {}
    for group in groups:
        role = group["canonical_role"]

        def load(suffix, role=role):
            name = role + "-" + suffix
            return _verified_bytes(directory / name, result["artifacts"][name])

        scores = np.load(io.BytesIO(load("scores.npy")), allow_pickle=False)
        predictions = [json.loads(line) for line in load("predictions.jsonl").splitlines()]
        metrics = json.loads(load("metrics.json"))
        audits[role] = verify_saved_evidence(rows, scores, predictions, metrics)
        for alias in group["roles"]:
            _require(result["metrics_by_role"][alias] == dict(
                metrics=metrics, canonical_role=role, identity=frozen["roles"][alias],
            ), "role summary or identity differs")
    return dict(
        status="PASS_INDEPENDENT_TRANSFER_OUTPUT_AUDIT", transfer_result_sha256=result_sha,
        comparison_sha256=comparison_sha, frozen_sha256=freeze_sha,
        groups=audits, **execution, model_loaded=False, optimizer_updates=0,
        calibration_loaded=False, sealed_audit_loaded=False, transfer_loaded=True,
        checkpoint_reselected=False, product_complete=False,
    ), checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--comparison-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, checked = inspect_transfer(
        args.directory, args.result_sha256, args.comparison, args.comparison_sha256
    )
    output = _outside_protected(checked, args.output)
    _require(
        not any(output.is_relative_to(_no_redirect(p)) for p in (args.directory, args.comparison)),
        "new audit output required",
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
