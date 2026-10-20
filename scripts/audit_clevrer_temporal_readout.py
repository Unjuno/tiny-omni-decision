"""Read-only data and saved-prediction audit for the CLEVRER temporal probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TEMPORAL_PROGRAM_GROUPS = {
    "collision_or_partner": {"filter_collision", "get_col_partner"},
    "entry_exit": {"filter_in", "filter_out"},
    "order": {"first", "last", "filter_order", "filter_before", "filter_after"},
    "frame": {"get_frame", "start", "end"},
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0, "accuracy": float("nan"), "nll": float("nan"), "brier": float("nan")}
    correct = 0
    nll = 0.0
    brier = 0.0
    for row in rows:
        options = row["options"]
        target = str(row["target"])
        probabilities = [float(item["probability"]) for item in options]
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError(f"invalid probability in prediction {row['id']}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError(f"probabilities do not sum to one in prediction {row['id']}")
        option_ids = [str(item["option_id"]) for item in options]
        if target not in option_ids:
            raise ValueError(f"target is absent from options in prediction {row['id']}")
        truth = option_ids.index(target)
        correct += str(row["prediction"]) == target
        nll -= math.log(max(probabilities[truth], 1e-12))
        brier += sum((p - (index == truth)) ** 2 for index, p in enumerate(probabilities))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "nll": nll / len(rows),
        "brier": brier / len(rows),
    }


def audit(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    if not train or not validation:
        raise ValueError("train and validation rows must be non-empty")
    for split_name, rows in (("train", train), ("validation", validation)):
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {split_name} sample IDs")
        for row in rows:
            options = [str(value) for value in row["options"]]
            if str(row["target"]) not in options or len(options) != len(set(options)):
                raise ValueError(f"invalid target/options at {row['id']}")
            if row.get("question_type") != "temporal_descriptive":
                raise ValueError(f"non-temporal row in temporal corpus: {row['id']}")

    train_scenes = {str(row["scene_group_id"]) for row in train}
    validation_scenes = {str(row["scene_group_id"]) for row in validation}
    train_media = {str(row["media_sha256"]) for row in train}
    validation_media = {str(row["media_sha256"]) for row in validation}

    def content(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
        return (
            _normal(str(row["question"])),
            tuple(sorted(_normal(str(x)) for x in row["options"])),
        )

    train_content = {content(row) for row in train}
    validation_content = {content(row) for row in validation}
    if (
        train_scenes & validation_scenes
        or train_media & validation_media
        or train_content & validation_content
    ):
        raise ValueError("cross-split scene, media, or normalized question/options overlap")

    prediction_by_id = {str(row["id"]): row for row in predictions}
    validation_by_id = {str(row["id"]): row for row in validation}
    if len(prediction_by_id) != len(predictions) or set(prediction_by_id) != set(validation_by_id):
        raise ValueError("saved prediction IDs must exactly match validation IDs")
    joined = []
    for sample_id, source in validation_by_id.items():
        prediction = prediction_by_id[sample_id]
        if str(prediction["target"]) != str(source["target"]):
            raise ValueError(f"saved target mismatch for {sample_id}")
        joined.append({**source, **prediction})

    train_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train:
        train_counts[str(row["taxonomy"])][str(row["target"])] += 1

    def groups(rows: list[dict[str, Any]], label: str, get_key: Any) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(get_key(row))].append(row)
        return {key: _metrics(value) for key, value in sorted(buckets.items())}

    prior_rows = []
    for row in validation:
        counts = train_counts[str(row["taxonomy"])]
        majority = min(counts, key=lambda label: (-counts[label], label))
        # The prior is a data diagnostic, not a replacement model baseline.
        total = sum(counts.values())
        options = [str(value) for value in row["options"]]
        probs = [counts.get(option, 0) / total for option in options]
        prior_rows.append(
            {
                **row,
                "prediction": majority,
                "options": [
                    {"option_id": option, "probability": probability}
                    for option, probability in zip(options, probs, strict=True)
                ],
            }
        )

    program_groups = {
        family: groups(
            joined,
            family,
            lambda row, tokens=tokens, family=family: (
                family if tokens.intersection(map(str, row["program"])) else "absent"
            ),
        )
        for family, tokens in TEMPORAL_PROGRAM_GROUPS.items()
    }
    # Keep the group attribution disjoint and explicit: a row may contain several operator families.
    operator_flags = {
        name: sum(bool(tokens.intersection(map(str, row["program"]))) for row in validation)
        for name, tokens in TEMPORAL_PROGRAM_GROUPS.items()
    }
    question_by_taxonomy: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train:
        question_by_taxonomy[str(row["taxonomy"])][_normal(str(row["question"]))] += 1

    return {
        "schema_version": 1,
        "scope": "read-only data audit and post-hoc diagnosis; no checkpoint/model selection",
        "counts": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_scenes": len(train_scenes),
            "validation_scenes": len(validation_scenes),
            "scene_overlap": 0,
            "media_overlap": 0,
            "normalized_question_options_overlap": 0,
            "validation_predictions_exact_id_match": True,
        },
        "taxonomy_counts": {
            split: dict(sorted(Counter(str(row["taxonomy"]) for row in rows).items()))
            for split, rows in (("train", train), ("validation", validation))
        },
        "train_target_counts": {
            taxonomy: dict(sorted(counts.items()))
            for taxonomy, counts in sorted(train_counts.items())
        },
        "unobserved_training_target_classes": {
            taxonomy: sorted(
                {
                    str(option)
                    for row in train
                    if row["taxonomy"] == taxonomy
                    for option in row["options"]
                }
                - set(counts)
            )
            for taxonomy, counts in sorted(train_counts.items())
        },
        "validation_target_counts": {
            taxonomy: dict(
                sorted(
                    Counter(
                        str(row["target"]) for row in validation if row["taxonomy"] == taxonomy
                    ).items()
                )
            )
            for taxonomy in sorted(train_counts)
        },
        "train_taxonomy_majority_prior_on_validation": _metrics(prior_rows),
        "saved_readout_validation": _metrics(joined),
        "saved_readout_by_taxonomy": groups(joined, "taxonomy", lambda row: row["taxonomy"]),
        "saved_readout_by_program_operator_presence": operator_flags,
        "saved_readout_by_operator_family": program_groups,
        "question_template_counts_by_taxonomy": {
            taxonomy: {
                "unique_normalized_questions": len(counter),
                "most_common": counter.most_common(10),
            }
            for taxonomy, counter in sorted(question_by_taxonomy.items())
        },
        "limitations": [
            "Validation results are post-hoc diagnostics on the development set used for "
            "checkpoint selection.",
            "CLEVRER question templates and typed programs are correlated; subgroup differences "
            "are descriptive.",
            "The audit checks corpus/prediction consistency, not physical correctness of videos "
            "or annotations.",
            "Readout and prior use the same fixed taxonomy option set for each validation row.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.corpus_dir / "train.jsonl"
    validation_path = args.corpus_dir / "validation.jsonl"
    train, validation, predictions = (
        _jsonl(train_path),
        _jsonl(validation_path),
        _jsonl(args.predictions),
    )
    result = audit(train, validation, predictions)
    result["input_sha256"] = {
        "train": _sha256(train_path),
        "validation": _sha256(validation_path),
        "predictions": _sha256(args.predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
