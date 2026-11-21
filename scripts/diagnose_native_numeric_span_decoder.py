"""TRAIN-only OOF diagnostic for reading numeric spans from frozen text features.

This uses gold JSON field/object spans as an explicit oracle diagnostic. It does
not test production parsing or end-to-end decision quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

FIELDS = ("cost", "risk", "time")
EXTREMA_TASKS = tuple(
    f"numeric:extremum:{field}:{direction}"
    for field in FIELDS
    for direction in ("minimum", "maximum")
)
VALUE_PATTERN = re.compile(r'"(cost|risk|time)"\s*:\s*(\d+)')
OBJECT_PATTERN = re.compile(r"\{[^{}]*\}")
SPLIT_ID = "native-numeric-span-ridge-cv-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_for_group(group: str, *, seed: int = 17, folds: int = 5) -> int:
    if not group or type(seed) is not int or seed < 0 or type(folds) is not int or folds < 2:
        raise ValueError("nonempty group, nonnegative seed, and at least two folds required")
    digest = hashlib.sha256(f"{seed}:{SPLIT_ID}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def numeric_span_features(
    state: str,
    offsets: list[tuple[int, int]],
    hidden: np.ndarray,
    *,
    prefix: str = "",
) -> list[dict]:
    """Pool frozen hidden states overlapping each numeric JSON value span."""
    if not isinstance(state, str) or not state or not isinstance(prefix, str):
        raise ValueError("nonempty JSON state and string prefix required")
    values = np.asarray(hidden)
    if values.ndim != 2 or values.shape[1] != 384 or values.shape[0] != len(offsets) + 2:
        raise ValueError("hidden states must include CLS/SEP around the 384-wide token sequence")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("hidden states must be finite float32")
    try:
        objects = json.loads(state)
    except json.JSONDecodeError as error:
        raise ValueError("state must contain valid JSON") from error
    matches = list(OBJECT_PATTERN.finditer(state))
    if not isinstance(objects, list) or len(objects) != len(matches) or not objects:
        raise ValueError("state JSON must be a list of objects with directly mappable spans")
    text_offset = len(prefix)
    output = []
    for object_index, (parsed, object_match) in enumerate(zip(objects, matches, strict=True)):
        if not isinstance(parsed, dict) or not isinstance(parsed.get("id"), str):
            raise ValueError("each state object needs a string ID")
        object_text = object_match.group(0)
        spans = list(VALUE_PATTERN.finditer(object_text))
        by_field = {match.group(1): match for match in spans}
        if set(by_field) != set(FIELDS) or len(spans) != len(FIELDS):
            raise ValueError("each object must contain exactly one cost, risk, and time")
        for field in FIELDS:
            match = by_field[field]
            numeric_value = int(match.group(2))
            if type(parsed.get(field)) is not int or parsed[field] != numeric_value:
                raise ValueError("JSON numeric span does not match parsed field value")
            start = text_offset + object_match.start() + match.start(2)
            stop = text_offset + object_match.start() + match.end(2)
            token_indices = [
                index
                for index, (token_start, token_stop) in enumerate(offsets)
                if token_start < stop and token_stop > start
            ]
            if not token_indices:
                raise ValueError("numeric value has no overlapping tokenizer spans")
            # The cached token array contains CLS at position zero and SEP at the end.
            feature = values[np.asarray(token_indices, dtype=np.int64) + 1].mean(axis=0)
            output.append(
                {
                    "item_id": parsed["id"],
                    "field": field,
                    "value": numeric_value,
                    "feature": np.asarray(feature, dtype=np.float32),
                    "object_index": object_index,
                    "token_piece_count": len(token_indices),
                }
            )
    if len(output) != len(objects) * len(FIELDS):
        raise AssertionError("numeric span extraction lost a field")
    return output


def ridge_oof_predictions(
    examples: list[dict], *, seed: int = 17, folds: int = 5, alpha: float = 1.0
) -> tuple[dict[tuple[str, str, str], float], dict[str, int]]:
    if not examples or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("nonempty examples and positive finite ridge alpha required")
    groups = sorted({row["group"] for row in examples})
    assignment = {group: fold_for_group(group, seed=seed, folds=folds) for group in groups}
    counts = Counter(assignment.values())
    if set(counts) != set(range(folds)):
        raise ValueError("group-CV assignment left at least one fold empty")
    if len({(row["group"], row["item_id"], row["field"]) for row in examples}) != len(examples):
        raise ValueError("duplicate group/item/field numeric feature")
    predictions: dict[tuple[str, str, str], float] = {}
    for fold in range(folds):
        train = [row for row in examples if assignment[row["group"]] != fold]
        test = [row for row in examples if assignment[row["group"]] == fold]
        train_groups = {row["group"] for row in train}
        test_groups = {row["group"] for row in test}
        if train_groups & test_groups or not train or not test:
            raise AssertionError("group cross-validation isolation failed")
        x_train = np.stack([row["feature"] for row in train]).astype(np.float64)
        y_train = np.asarray([row["value"] for row in train], dtype=np.float64)
        x_test = np.stack([row["feature"] for row in test]).astype(np.float64)
        x_mean = x_train.mean(axis=0)
        x_scale = x_train.std(axis=0)
        x_scale[x_scale < 1e-8] = 1.0
        y_mean = float(y_train.mean())
        y_scale = float(y_train.std())
        if y_scale < 1e-8:
            raise ValueError("numeric target has no training variation")
        x_train = (x_train - x_mean) / x_scale
        x_test = (x_test - x_mean) / x_scale
        y_train = (y_train - y_mean) / y_scale
        gram = x_train.T @ x_train
        gram.flat[:: gram.shape[0] + 1] += alpha
        coefficient = np.linalg.solve(gram, x_train.T @ y_train)
        y_hat = ((x_test @ coefficient) * y_scale) + y_mean
        for row, value in zip(test, y_hat, strict=True):
            predictions[(row["group"], row["item_id"], row["field"])] = float(value)
    if len(predictions) != len(examples):
        raise AssertionError("OOF predictions do not cover each numeric span exactly once")
    return predictions, assignment


def deranged_prediction_control(
    predictions: dict[tuple[str, str, str], float],
    *,
    groups: dict[str, list[str]],
    seed: int = 17,
) -> dict[tuple[str, str, str], float]:
    control = {}
    for group, item_ids in sorted(groups.items()):
        if len(item_ids) < 2 or len(item_ids) != len(set(item_ids)):
            raise ValueError("each control group needs distinct candidate IDs")
        for field in FIELDS:
            ordered = sorted(
                item_ids,
                key=lambda item: hashlib.sha256(
                    f"{seed}:{SPLIT_ID}:shuffle:{group}:{field}:{item}".encode()
                ).digest(),
            )
            donors = {
                item: ordered[(index + 1) % len(ordered)] for index, item in enumerate(ordered)
            }
            if any(item == donor for item, donor in donors.items()):
                raise AssertionError("feature-to-candidate shuffle is not a derangement")
            for item, donor in donors.items():
                control[(group, item, field)] = predictions[(group, donor, field)]
    return control


def score_extrema(groups: dict[str, dict], predictions: dict[tuple[str, str, str], float]) -> dict:
    per_task = {task: [] for task in EXTREMA_TASKS}
    per_group = {}
    for group, payload in sorted(groups.items()):
        objects = payload["objects"]
        item_ids = [item["id"] for item in objects]
        task_correct = []
        for task in EXTREMA_TASKS:
            _, _, field, direction = task.split(":")
            row = payload["tasks"].get(task)
            if row is None:
                raise ValueError(f"missing fixed task row for {group}/{task}")
            order = row["option_ids"]
            if set(order) != set(item_ids) or row["target_id"] not in item_ids:
                raise ValueError("task option IDs do not match the parsed state objects")
            position = {item: index for index, item in enumerate(order)}
            pred_id = min(
                item_ids,
                key=lambda item: (
                    predictions[(group, item, field)] * (1 if direction == "minimum" else -1),
                    position[item],
                ),
            )
            correct = int(pred_id == row["target_id"])
            per_task[task].append(correct)
            task_correct.append(correct)
        per_group[group] = float(np.mean(task_correct))
    task_accuracy = {task: float(np.mean(values)) for task, values in per_task.items()}
    return {
        "accuracy_by_task": task_accuracy,
        "macro_accuracy": float(np.mean(list(task_accuracy.values()))),
        "overall_accuracy": float(
            np.mean([value for values in per_task.values() for value in values])
        ),
        "correct_by_group": per_group,
        "group_count": len(per_group),
    }


def percentile_group_bootstrap(
    left: dict[str, float], right: dict[str, float], *, seed: int = 17, draws: int = 5000
) -> dict:
    groups = sorted(left)
    if groups != sorted(right) or len(groups) < 2 or draws < 100:
        raise ValueError("paired group vectors and at least 100 bootstrap draws required")
    left_values = np.asarray([left[group] for group in groups], dtype=np.float64)
    right_values = np.asarray([right[group] for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    deltas = (left_values - right_values)[indices].mean(axis=1)
    return {
        "method": "paired percentile bootstrap over whole state groups",
        "draws": draws,
        "seed": seed,
        "delta_mean": float((left_values - right_values).mean()),
        "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--language-config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--numeric-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {args.output}")

    config = json.loads(args.diagnostic_config.read_text(encoding="utf-8"))
    language_config = json.loads(args.language_config.read_text(encoding="utf-8"))
    expected = {
        str(args.numeric_train): language_config["input_pins"][str(args.numeric_train)],
        str(args.split_manifest): config["split_manifest_sha256"],
    }
    corpus_root = Path(config["corpus_root"])
    for name, digest in config["corpus_input_pins"].items():
        expected[str(corpus_root / name)] = digest
    state_bank_path = Path(config["text_state_bank_path"])
    expected[str(state_bank_path)] = config["text_state_bank_sha256"]
    model_path = Path(language_config["model_path"])
    for name, digest in config["tokenizer_file_sha256"].items():
        expected[str(model_path / name)] = digest
    for path, digest in expected.items():
        if not Path(path).is_file() or sha256_file(Path(path)) != digest:
            raise ValueError(f"pinned input hash mismatch: {path}")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    fit_ids = set(split.get("fit_ids", []))
    fit_groups = set(split.get("fit_groups", []))
    if len(fit_ids) != 6144 or len(fit_groups) != 768:
        raise ValueError("the fixed diagnostic requires exactly 6,144 FIT IDs / 768 FIT groups")
    if split.get("split_id") != "native-text-programmatic-group-holdout-v1":
        raise ValueError("unexpected prior FIT split identity")

    raw_by_group: dict[str, dict] = {}
    raw_ids = set()
    task_counts = Counter()
    with args.numeric_train.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_id = "text:" + row.get("id", "")
            if row_id not in fit_ids:
                continue
            if row.get("source") != "programmatic_decision_v1" or row.get("split") != "train":
                raise ValueError("FIT IDs must resolve only to programmatic official TRAIN rows")
            group = row["group"]
            if group not in fit_groups or not row.get("state"):
                raise ValueError("FIT row group/state is outside the frozen FIT set")
            payload = raw_by_group.setdefault(group, {"state": row["state"], "tasks": {}})
            if payload["state"] != row["state"]:
                raise ValueError("one state group maps to multiple raw state strings")
            task = row["task"]
            if task in payload["tasks"]:
                raise ValueError("duplicate task row inside one FIT group")
            payload["tasks"][task] = {
                "target_id": row["target_id"],
                "option_ids": list(row["option_ids"]),
            }
            raw_ids.add(row_id)
            task_counts[task] += 1
    if raw_ids != fit_ids or set(raw_by_group) != fit_groups:
        raise ValueError("the frozen FIT split does not resolve to exact source rows")

    standard_metadata = {}
    train_path = corpus_root / "train.jsonl"
    with train_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("id") not in fit_ids:
                continue
            if row.get("source") != "text:programmatic_decision_v1" or row.get("split") != "train":
                raise ValueError("normalized FIT rows must remain official TRAIN text rows")
            group = row["group"]
            if group not in fit_groups:
                raise ValueError("normalized FIT row group is outside the frozen FIT set")
            prior = standard_metadata.setdefault(group, row["observation"])
            if prior != row["observation"]:
                raise ValueError("one state group maps to multiple cached observations")
    if set(standard_metadata) != fit_groups:
        raise ValueError("normalized frozen TRAIN corpus is missing FIT groups")
    if any(task_counts[task] != 768 for task in task_counts):
        raise ValueError("each source task must have one row per FIT state group")
    for _group, payload in raw_by_group.items():
        if set(payload["tasks"]) != set(EXTREMA_TASKS) | {
            "numeric:count",
            "numeric:filter_min_cost",
        }:
            raise ValueError("FIT group is missing its fixed numeric task rows")
        objects = json.loads(payload["state"])
        if len(objects) != 4 or len({item.get("id") for item in objects}) != 4:
            raise ValueError("each numeric state must have four uniquely identified items")
        for task in EXTREMA_TASKS:
            _, _, field, direction = task.split(":")
            target = (
                min(objects, key=lambda item: item[field])["id"]
                if direction == "minimum"
                else max(objects, key=lambda item: item[field])["id"]
            )
            if payload["tasks"][task]["target_id"] != target:
                raise ValueError(
                    "gold TRAIN task target disagrees with raw deterministic state rule"
                )
        payload["objects"] = objects

    observations = json.loads((corpus_root / "observations.json").read_text(encoding="utf-8"))
    bank = np.load(state_bank_path, mmap_mode="r", allow_pickle=False)
    if bank.shape != (375078, 384) or bank.dtype != np.float32:
        raise ValueError("cached frozen state bank shape/dtype differs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer with character offsets is required")
    examples = []
    for group in sorted(fit_groups):
        state = raw_by_group[group]["state"]
        observation_id = standard_metadata[group]
        observation = observations.get(observation_id)
        if (
            not observation
            or observation.get("split") != "train"
            or observation.get("source") != "text"
        ):
            raise ValueError("FIT feature mapping does not resolve to TRAIN text state")
        features = observation.get("features", [])
        if len(features) != 1 or features[0].get("bank") != "language/train-states.npy":
            raise ValueError("FIT state feature mapping differs from pinned TRAIN bank")
        offset, tokens = int(features[0]["offset"]), int(features[0]["tokens"])
        text = state
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
        if len(offsets) + 2 != tokens:
            raise ValueError("FIT text tokenization no longer matches the cached frozen span")
        hidden = np.asarray(bank[offset : offset + tokens])
        if hidden.shape != (tokens, 384) or not np.isfinite(hidden).all():
            raise ValueError("invalid pinned FIT feature span")
        extracted = numeric_span_features(state, offsets, hidden)
        if len(extracted) != 12:
            raise ValueError("each FIT state must contribute twelve numeric values")
        for example in extracted:
            examples.append({"group": group, **example})
        payload = raw_by_group[group]
        payload["objects"] = json.loads(state)

    predictions, assignment = ridge_oof_predictions(examples, seed=17, folds=5, alpha=1.0)
    shuffled = deranged_prediction_control(
        predictions,
        groups={
            group: [item["id"] for item in raw_by_group[group]["objects"]] for group in fit_groups
        },
        seed=17,
    )
    score_groups = {
        group: {
            "objects": payload["objects"],
            "tasks": payload["tasks"],
        }
        for group, payload in raw_by_group.items()
    }
    model_scores = score_extrema(score_groups, predictions)
    shuffle_scores = score_extrema(score_groups, shuffled)
    raw_oracle = score_extrema(
        score_groups,
        {
            (group, item["id"], field): float(item[field])
            for group, payload in raw_by_group.items()
            for item in payload["objects"]
            for field in FIELDS
        },
    )

    y_true = np.asarray([row["value"] for row in examples], dtype=np.float64)
    y_pred = np.asarray(
        [predictions[(row["group"], row["item_id"], row["field"])] for row in examples],
        dtype=np.float64,
    )
    regression = {}
    for field in FIELDS:
        indices = np.asarray([row["field"] == field for row in examples], dtype=bool)
        truth, estimate = y_true[indices], y_pred[indices]
        regression[field] = {
            "value_count": int(indices.sum()),
            "mae": float(np.mean(np.abs(truth - estimate))),
            "rmse": float(np.sqrt(np.mean((truth - estimate) ** 2))),
            "pearson_r": float(np.corrcoef(truth, estimate)[0, 1]),
        }
    regression["macro"] = {
        "value_count": len(examples),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }
    bootstrap = percentile_group_bootstrap(
        model_scores["correct_by_group"], shuffle_scores["correct_by_group"], seed=17, draws=5000
    )

    result = {
        "schema_version": 1,
        "status": "PASS_TRAIN_ONLY_GROUP_OOF_NUMERIC_SPAN_DIAGNOSTIC",
        "started_utc": started_utc,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "config_sha256": sha256_file(args.diagnostic_config),
        "language_config_sha256": sha256_file(args.language_config),
        "input_sha256": expected,
        "tokenizer": {
            "path": str(model_path),
            "revision": language_config["model_revision"],
            "fast": tokenizer.is_fast,
            "files": config["tokenizer_file_sha256"],
        },
        "protocol": {
            "source": "text:programmatic_decision_v1 official TRAIN only",
            "cached_observation_text": "raw JSON state, matching the pinned native feature bank",
            "fit_group_count": len(fit_groups),
            "fit_row_count": len(fit_ids),
            "numeric_span_count": len(examples),
            "group_cv_folds": 5,
            "fold_assignment": f"SHA256(seed=17:{SPLIT_ID}:group) modulo 5",
            "ridge_alpha": 1.0,
            "field_identity": "gold JSON key span supplied to diagnostic decoder",
            "object_to_candidate_mapping": "gold JSON object ID supplied to diagnostic scorer",
            "decision_direction": "gold task specifies minimum or maximum",
            "official_validation_loaded": False,
            "prior_fit_holdout_rows_or_labels_used": False,
            "prior_fit_holdout_predictions_or_metrics_loaded": False,
            "sealed_audit_loaded": False,
            "gpu_used": False,
            "cloud_cost_usd": 0,
        },
        "fold_group_counts": dict(sorted(Counter(assignment.values()).items())),
        "regression": regression,
        "extrema_oof": {
            "ridge": {
                key: value for key, value in model_scores.items() if key != "correct_by_group"
            },
            "within_group_feature_derangement": {
                key: value for key, value in shuffle_scores.items() if key != "correct_by_group"
            },
            "raw_numeric_oracle": {
                key: value for key, value in raw_oracle.items() if key != "correct_by_group"
            },
            "paired_group_bootstrap_ridge_minus_derangement": bootstrap,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output / "fit-oof-numeric-predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in sorted(
            examples, key=lambda item: (item["group"], item["field"], item["item_id"])
        ):
            key = (row["group"], row["item_id"], row["field"])
            stream.write(
                json.dumps(
                    {
                        "group": row["group"],
                        "item_id": row["item_id"],
                        "field": row["field"],
                        "target_value": row["value"],
                        "oof_value_prediction": predictions[key],
                        "cv_fold": assignment[row["group"]],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    result["prediction_file_sha256"] = sha256_file(prediction_path)
    result["finished_utc"] = datetime.now(UTC).isoformat()
    result["elapsed_seconds"] = time.perf_counter() - started
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    result["summary_sha256"] = sha256_file(summary_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
