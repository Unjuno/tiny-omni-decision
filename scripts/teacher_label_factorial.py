"""Separate candidate display order from label assignment on fixed TRAIN questions."""

from collections import defaultdict

import numpy as np

from scripts.semantic_teacher_probe import LABELS, build_probe, probabilities, summarize_rows

ARMS = ("00", "10", "01", "11")  # display reversal, semantic-ID label-map reversal


def build_factorial(binding, numeric, count=32):
    old = build_probe(binding, numeric, count)
    result = []
    for base in old:
        if base["condition"] not in ("exclusion", "original"):
            continue
        canonical_ids = base["option_ids"]
        for arm in ARMS:
            order, mapping = map(int, arm)
            ids = canonical_ids[::-1] if order else canonical_ids.copy()
            labels = list(LABELS[: len(ids)])
            label_map = dict(zip(canonical_ids, labels[::-1] if mapping else labels, strict=True))
            descriptions = dict(zip(canonical_ids, base["options"], strict=True))
            condition = {
                "00": base["condition"],
                "10": "order_only",
                "01": "labels_only",
                "11": "reverse_exclusion" if base["source"] == "binding" else "reverse",
            }[arm]
            baseline_id = None
            if arm == "00":
                baseline_id = base["id"]
            elif arm == "11":
                baseline_id = (
                    base["id"].replace(":exclusion:", ":reverse_exclusion:")
                    if base["source"] == "binding"
                    else base["id"].removesuffix(":original") + ":reverse"
                )
            result.append(
                dict(
                    base,
                    id=f"factorial:{base['id']}:{arm}",
                    base_id=base["id"],
                    arm=arm,
                    condition=condition,
                    baseline_row_id=baseline_id,
                    option_ids=ids,
                    options=[descriptions[oid] for oid in ids],
                    labels=[label_map[oid] for oid in ids],
                )
            )
    return result


def render_input(row):
    labels, options = row["labels"], row["options"]
    if (
        len(labels) != len(options)
        or set(labels) != set(LABELS[: len(options)])
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("complete distinct one-character labels required")
    choices = "\n".join(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
    return (
        f"State: {row['state']}\nQuestion: {row['question']}\nOptions:\n{choices}\n"
        "Choose the best option. Return only its case-sensitive option label, without explanation."
    )


def summarize_factorial_rows(rows, predictions):
    # The legacy metric helper pairs display reversal by its legacy row IDs.
    # Adapt only this local metric view; saved input and prediction IDs are unchanged.
    if len(rows) != len(predictions):
        raise ValueError("complete aligned rows required")
    view_rows, view_predictions = [], []
    for row, prediction in zip(rows, predictions, strict=True):
        if row["id"] != prediction["row_id"]:
            raise ValueError("prediction identity mismatch")
        if row["source"] == "binding":
            identity = row["base_id"].replace(":exclusion:", f":{row['condition']}:")
        else:
            identity = row["base_id"].removesuffix(":original") + f":{row['condition']}"
        view_rows.append(dict(row, id=identity))
        view_predictions.append(dict(prediction, row_id=identity))
    return summarize_rows(view_rows, view_predictions)


def factorial_summary(rows, predictions):
    if len(rows) != len(predictions) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("complete unique rows required")
    blocks = defaultdict(dict)
    for row, pred in zip(rows, predictions, strict=True):
        if row["id"] != pred["row_id"] or len(pred["logits"]) != len(row["options"]):
            raise ValueError("aligned output required")
        p = probabilities(pred["logits"])
        target = row["option_ids"].index(row["target_id"])
        y = np.zeros(len(p))
        y[target] = 1
        item = dict(
            row=row,
            p=dict(zip(row["option_ids"], p, strict=True)),
            prediction=row["option_ids"][int(p.argmax())],
            scores=np.array(
                [
                    float(p.argmax() == target),
                    -np.log(max(p[target], 1e-12)),
                    np.square(p - y).sum(),
                ]
            ),
        )
        if row["arm"] in blocks[(row["source"], row["base_id"])]:
            raise ValueError("duplicate factorial arm")
        blocks[(row["source"], row["base_id"])][row["arm"]] = item
    if any(set(block) != set(ARMS) for block in blocks.values()):
        raise ValueError("all four arms required")
    contrasts = {
        "order_with_original_mapping": {"10": 1, "00": -1},
        "order_with_reversed_mapping": {"11": 1, "01": -1},
        "mapping_with_original_order": {"01": 1, "00": -1},
        "mapping_with_reversed_order": {"11": 1, "10": -1},
        "interaction": {"11": 1, "10": -1, "01": -1, "00": 1},
    }
    pairs = (("00", "10"), ("00", "01"), ("01", "11"), ("10", "11"))
    report = {}
    for source in ("binding", "numeric"):
        source_blocks = [b for (s, _), b in blocks.items() if s == source]
        grouped = defaultdict(list)
        for block in source_blocks:
            grouped[block["00"]["row"]["group"]].append(block)
        if not grouped:
            raise ValueError("both fixed sources required")
        groups = [grouped[g] for g in sorted(grouped)]
        rng = np.random.default_rng(17)
        draws = rng.integers(0, len(groups), (10000, len(groups)))
        result = dict(
            groups=len(groups), question_count=len(source_blocks), contrasts={}, invariance={}
        )
        for name, coefficients in contrasts.items():
            values = np.array(
                [
                    np.mean(
                        [
                            sum(coefficients[a] * b[a]["scores"] for a in coefficients)
                            for b in group
                        ],
                        axis=0,
                    )
                    for group in groups
                ]
            )
            boot = values[draws].mean(1)
            result["contrasts"][name] = dict(
                metrics=["accuracy", "nll", "brier"],
                point=values.mean(0).tolist(),
                lower=np.quantile(boot, 0.025, axis=0).tolist(),
                upper=np.quantile(boot, 0.975, axis=0).tolist(),
                draws=10000,
                seed=17,
                method="paired TRAIN-state percentile bootstrap; not independent/blind quality",
            )
        for a, b in pairs:
            differences = [
                max(abs(block[a]["p"][oid] - block[b]["p"][oid]) for oid in block[a]["p"])
                for block in source_blocks
            ]
            result["invariance"][f"{a}->{b}"] = dict(
                answer_agreement=float(
                    np.mean(
                        [
                            block[a]["prediction"] == block[b]["prediction"]
                            for block in source_blocks
                        ]
                    )
                ),
                maximum_probability_difference=float(max(differences)),
            )
        report[source] = result
    return report
