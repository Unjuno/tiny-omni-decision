"""Independent saved-product audit: no project imports, torch, fit or audit data."""

import argparse
import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    if isinstance(a, dict):
        need(isinstance(b, dict) and set(a) == set(b), "dictionary coverage")
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, list):
        need(isinstance(b, list) and len(a) == len(b), "list coverage")
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        need(np.isfinite(a) and np.isfinite(b), "nonfinite comparison")
        need(abs(a - b) <= 1e-12, f"metric mismatch {a} != {b}")
    else:
        need(a == b, f"value mismatch {a} != {b}")


def probabilities(scores):
    scores = np.asarray(scores, dtype=np.float64)
    need(scores.ndim >= 2 and np.isfinite(scores).all(), "finite scores")
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def measure(scores, target):
    scores, target = np.asarray(scores), np.asarray(target)
    need(scores.ndim == 2 and target.shape == (len(scores),), "metric dimensions")
    need(len(scores) > 0 and np.issubdtype(target.dtype, np.integer), "metric labels")
    need((target >= 0).all() and (target < scores.shape[1]).all(), "target range")
    p = probabilities(scores)
    correct = scores.argmax(1) == target
    confidence = p.max(1)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(p)), target] = 1
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        selected = (confidence >= lower) & ((confidence < upper) if index < 14 else True)
        if selected.any():
            ece += abs(float(correct[selected].sum()) - float(confidence[selected].sum())) / len(p)
    return dict(
        count=len(p),
        accuracy=float(correct.mean()),
        nll=float(-np.log(p[np.arange(len(p)), target]).mean()),
        brier=float(np.square(p - one_hot).sum(axis=1).mean()),
        ece=ece,
        mean_confidence=float(confidence.mean()),
    )


def summary(scores, target, fields):
    need(scores.shape == (len(target), 4, 40), "full candidate scores")
    correct = scores.argmax(2) == target
    per = {field: measure(scores[:, i], target[:, i]) for i, field in enumerate(fields)}
    both = {
        f"{fields[a]}+{fields[b]}": float((correct[:, a] & correct[:, b]).mean())
        for a, b in itertools.combinations(range(4), 2)
    }
    restricted = [
        (scores[:, q, q * 10 : (q + 1) * 10].argmax(1) == target[:, q] % 10).mean()
        for q in range(4)
    ]
    return dict(
        overall=measure(scores.reshape(-1, 40), target.reshape(-1)),
        per_question=per,
        macro={
            key: float(np.mean([v[key] for v in per.values()]))
            for key in ("accuracy", "nll", "brier", "ece")
        },
        all_four_correct=float(correct.all(1).mean()),
        changed_question_pair_both_correct=float(np.mean(list(both.values()))),
        pair_count=len(target) * 6,
        per_pair_both_correct=both,
        field_routing_accuracy=float((scores.argmax(2) // 10 == np.arange(4)).mean()),
        within_correct_field_accuracy=float(np.mean(restricted)),
    )


def bootstrap(before, after, target, groups):
    delta = (after.argmax(2) == target).astype(int) - (before.argmax(2) == target)
    cluster_indices = [np.flatnonzero(np.asarray(groups) == g) for g in sorted(set(groups))]
    rng = np.random.default_rng(17)
    values = []
    for _ in range(10000):
        sampled = np.concatenate(
            [
                cluster_indices[i]
                for i in rng.integers(len(cluster_indices), size=len(cluster_indices))
            ]
        )
        values.append(delta[sampled].mean())
    return dict(
        baseline_accuracy=float((before.argmax(2) == target).mean()),
        candidate_accuracy=float((after.argmax(2) == target).mean()),
        difference=float(delta.mean()),
        interval95=np.quantile(values, [0.025, 0.975], method="linear").tolist(),
        clusters=len(cluster_indices),
        images=len(target),
        questions=int(target.size),
        replicates=10000,
        seed=17,
        method="paired whole-annotation-state cluster percentile; image/question-weighted ratio",
        blind_evaluation=False,
        observed_train_diagnostic=True,
    )


def validate_folder(folder, expected_hash):
    result = read(folder / "result.json")
    need(sha(folder / "result.json") == expected_hash, "result hash")
    need(result["status"] == "PASS_FIXED_SEMANTIC_NATIVE_COMPARISON", "terminal status")
    need(
        not result["smoke"] and result["images"] == 256 and result["questions"] == 1024,
        "full cohort",
    )
    need(result["actual_candidate_pairs"] == 185040, "forward accounting")
    need(result["optimizer_updates"] == 0 and result["all_model_parameters_frozen"], "frozen")
    need(not result["transfer_loaded"] and not result["sealed_audit_loaded"], "no holdout")
    need(
        result["aliases"] == {"step0": "step0", "endpoint": "endpoint", "selected": "step0"},
        "selection aliases",
    )
    need(read(folder / "active.json")["phase"] == "complete", "terminal marker")
    for pins in [result["input_pins"], result["code_pins"], *result["consumed_dependency_pins"]]:
        for path, digest in pins.items():
            need(sha(path) == digest, "consumed source changed: " + path)
    expected_files = {
        "effective-config.json",
        "selection.json",
        "states.json",
        "targets.npy",
        "native-logits.npy",
        "metrics.json",
    }
    conditions = ("oracle", "predicted", "no_evidence", "no_question", "donor")
    for arm in ("step0", "endpoint"):
        for condition in conditions:
            expected_files.update(
                {f"{arm}-{condition}-scores.npy", f"{arm}-{condition}-predictions.jsonl"}
            )
    need(set(result["artifacts"]) == expected_files, "exact 26-payload coverage")
    for name, digest in result["artifacts"].items():
        need(sha(folder / name) == digest, "changed output: " + name)
    effective = read(folder / "effective-config.json")
    old_root = Path(effective["config"]["artifact_root"]) / "semantic-state-bridge-01"
    for name in ("selection.json", "states.json", "targets.npy", "native-logits.npy"):
        need(
            sha(folder / name) == sha(old_root / name), "native inputs differ from historical probe"
        )
    old_effective = read(old_root / "effective-config.json")
    equal(effective["questions"], old_effective["questions"])
    equal(effective["options"], old_effective["options"])
    need(effective["selected_step"] == 0 and effective["parameters"] == 70830722, "model scope")
    need(effective["trainable_parameters"] == 0 and effective["dtype"] == "float32", "inference")
    selection, states = read(folder / "selection.json"), read(folder / "states.json")
    target = np.load(folder / "targets.npy", allow_pickle=False)
    native = np.load(folder / "native-logits.npy", allow_pickle=False)
    option_ids = [x["id"] for x in effective["options"]]
    fields = [option_ids[q * 10].split(":", 1)[0] for q in range(4)]
    need(len(set(option_ids)) == 40 and target.shape == (256, 4), "candidate/label shape")
    need(np.array_equal(target // 10, np.broadcast_to(np.arange(4), target.shape)), "field targets")
    need(len({r["label"]["id"] for r in selection}) == 256, "unique images")
    need(len({r["media"]["sha256"] for r in selection}) == 256, "unique media")
    groups = []
    for i, row in enumerate(selection):
        need(row["label"]["split"] == row["media"]["split"] == "train", "train only")
        groups.append(
            hashlib.sha256(
                json.dumps(states["oracle"][i], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        for q, field in enumerate(fields):
            value = str(row["label"][field])
            need(states["oracle"][i][field] == value, "oracle label")
            need(option_ids[target[i, q]] == field + ":" + value, "target identity")
            inferred = option_ids[q * 10 + native[i, q].argmax()].split(":", 1)[1]
            need(states["predicted"][i][field] == inferred, "native predicted fact")
    need(Counter(Counter(groups).values()) == {1: 248, 2: 4}, "whole state clusters")
    donor_target = np.roll(target, -1, axis=0)
    need(int((target != donor_target).sum()) == 912, "donor changed labels")
    stored = read(folder / "metrics.json")
    all_arrays = {}
    verified_predictions = 0
    for arm in ("step0", "endpoint"):
        arrays = {
            kind: np.load(folder / f"{arm}-{kind}-scores.npy", allow_pickle=False)
            for kind in conditions
        }
        need(np.array_equal(arrays["donor"], np.roll(arrays["oracle"], -1, axis=0)), "donor scores")
        need(
            np.array_equal(arrays["no_evidence"], np.repeat(arrays["no_evidence"][:1], 256, 0)),
            "empty evidence reuse",
        )
        need(
            np.array_equal(arrays["no_question"], np.repeat(arrays["no_question"][:, :1], 4, 1)),
            "empty question reuse",
        )
        calculated = {kind: summary(score, target, fields) for kind, score in arrays.items()}
        calculated["donor_corrected"] = summary(arrays["donor"], donor_target, fields)
        calculated["donor_target_changed_questions"] = 912
        truth_native = native.argmax(2) == target % 10
        truth_oracle = arrays["oracle"].argmax(2) == target
        truth_predicted = arrays["predicted"].argmax(2) == target
        intersections = Counter(
            f"native_{int(a)}_oracle_{int(b)}_predicted_{int(c)}"
            for a, b, c in zip(
                truth_native.flat, truth_oracle.flat, truth_predicted.flat, strict=True
            )
        )
        calculated["error_attribution"] = dict(
            questions=1024,
            native_fact_accuracy=float(truth_native.mean()),
            oracle_decision_accuracy=float(truth_oracle.mean()),
            predicted_decision_accuracy=float(truth_predicted.mean()),
            intersections=dict(intersections),
        )
        equal(calculated, stored["arms"][arm])
        need(calculated["no_question"]["overall"]["accuracy"] <= 0.25, "no question ceiling")
        need(
            calculated["no_question"]["changed_question_pair_both_correct"] == 0,
            "no question pairs",
        )
        for kind, score in arrays.items():
            p = probabilities(score)
            with (folder / f"{arm}-{kind}-predictions.jsonl").open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            need(len(rows) == 1024, "prediction coverage")
            for j, row in enumerate(rows):
                i, q = divmod(j, 4)
                need(
                    row["id"] == selection[i]["label"]["id"] + ":" + fields[q],
                    "prediction ID/order",
                )
                need(row["state_group"] == groups[i], "prediction state grouping")
                need(
                    row["option_ids"] == option_ids and row["target"] == int(target[i, q]),
                    "prediction labels",
                )
                need(np.array_equal(row["logits"], score[i, q]), "saved logits")
                need(
                    np.allclose(row["probabilities"], p[i, q], atol=1e-14, rtol=0),
                    "saved probabilities",
                )
                need(row["prediction"] == int(score[i, q].argmax()), "prediction argmax")
                need(row["predicted_id"] == option_ids[row["prediction"]], "predicted option ID")
                if kind == "donor":
                    need(row["donor_corrected_target"] == int(donor_target[i, q]), "donor label")
                verified_predictions += 1
        check = stored["controls"][arm]
        need(check["predictions_exact"] and check["all_candidates_retained"], "order controls")
        need(
            max(check["permutation_score_max_abs"], check["singleton_score_max_abs"]) <= 1e-5,
            "numeric controls",
        )
        all_arrays[arm] = arrays
    for kind in ("oracle", "predicted"):
        equal(
            bootstrap(all_arrays["step0"][kind], all_arrays["endpoint"][kind], target, groups),
            stored["endpoint_minus_step0"][kind],
        )
    equal(
        {field: measure(native[:, q], target[:, q] % 10) for q, field in enumerate(fields)},
        stored["native_ten_option"],
    )
    equal(stored["aliases"], result["aliases"])
    return result, dict(
        predictions=verified_predictions,
        whole_state_groups=252,
        payloads=len(expected_files),
        full_candidate_choices=40,
        metrics_recomputed=True,
        both_bootstraps_recomputed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--primary-sha256", required=True)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    need(not args.output.exists(), "new output required")
    primary, checked = validate_folder(args.primary, args.primary_sha256)
    replay, replay_checked = validate_folder(args.replay, args.replay_sha256)
    need(primary["pid"] != replay["pid"], "distinct process required")
    need(
        datetime.fromisoformat(primary["finished_utc"])
        < datetime.fromisoformat(replay["started_utc"]),
        "sequential full replay required",
    )
    for name in (
        "source_commit",
        "code_pins",
        "input_pins",
        "consumed_dependency_pins",
        "artifacts",
    ):
        equal(primary[name], replay[name])
    equal(sorted(replay["fresh_replay_files_exact"]), sorted(primary["artifacts"]))
    output = dict(
        status="PASS_INDEPENDENT_SEMANTIC_NATIVE_VERIFICATION",
        primary_sha256=args.primary_sha256,
        replay_sha256=args.replay_sha256,
        script_sha256=sha(__file__),
        primary=checked,
        replay=replay_checked,
        selected_step=0,
        optimizer_updates=0,
        observed_train_diagnostic=True,
        product_complete=False,
        sealed_audit_loaded=False,
        transfer_loaded=False,
    )
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
