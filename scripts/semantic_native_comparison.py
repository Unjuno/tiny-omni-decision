"""Pure helpers for one observed native-image diagnosis, never a product router."""

import json
from collections import Counter

import numpy as np


def replay_file_names(actual_names, expected_hashes):
    if not expected_hashes or set(actual_names) != set(expected_hashes):
        raise ValueError("fresh replay payload-name coverage mismatch")
    return sorted(expected_hashes)


def compare_public_pairs(row):
    state, question, options = row["state"], row["question"], row["options"]
    if (
        not isinstance(state, dict)
        or (state and set(state) != {"color", "texture", "count", "shape"})
        or any(not isinstance(v, str) for v in state.values())
        or not isinstance(question, str)
        or not isinstance(options, list)
        or len(options) < 2
        or any(not isinstance(v, str) or not v for v in options)
        or len(set(options)) != len(options)
    ):
        raise ValueError("public four-fact state, question and distinct candidate texts required")
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return [
        (f"Question: {question}\nCandidate answer: {option}", f"State: {encoded}")
        for option in options
    ]


def checked_scores(before, after, target):
    before, after, target = np.asarray(before), np.asarray(after), np.asarray(target)
    if (
        before.shape != after.shape
        or before.ndim != 3
        or before.shape[1:] != (4, 40)
        or len(before) == 0
        or target.shape != before.shape[:2]
        or not np.issubdtype(target.dtype, np.integer)
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or not np.array_equal(target // 10, np.broadcast_to(np.arange(4), target.shape))
    ):
        raise ValueError("aligned finite image x four-question x forty-option scores required")
    return before, after, target


def permutation_check(original, raw_permuted, permutation, tolerance):
    original, raw_permuted = np.asarray(original), np.asarray(raw_permuted)
    permutation = np.asarray(permutation)
    if (
        original.shape != raw_permuted.shape
        or original.ndim != 3
        or original.shape[1:] != (4, 40)
        or len(original) == 0
        or permutation.shape != (40,)
        or not np.issubdtype(permutation.dtype, np.integer)
        or sorted(permutation.tolist()) != list(range(40))
        or not np.isfinite(original).all()
        or not np.isfinite(raw_permuted).all()
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("aligned finite probe scores and complete option permutation required")
    restored = raw_permuted[:, :, np.argsort(permutation)]
    difference = float(abs(restored - original).max())
    # Predict in the displayed order first. Reordering before argmax hides ties.
    actual_ids = permutation[raw_permuted.argmax(-1)]
    exact = np.array_equal(actual_ids, original.argmax(-1))
    if difference > tolerance or not exact:
        raise ValueError("candidate-order numerical or actual predicted-ID mismatch")
    return dict(max_abs=difference, predicted_option_ids_exact=True)


def paired_accuracy_interval(before, after, target, groups, replicates=10000, seed=17):
    before, after, target = checked_scores(before, after, target)
    if len(groups) != len(target) or replicates < 1 or any(not isinstance(g, str) for g in groups):
        raise ValueError("one state-cluster identity per image and positive repetitions required")
    difference = (after.argmax(-1) == target).astype(int) - (before.argmax(-1) == target)
    clusters = sorted(set(groups))
    sums, sizes = [], []
    for cluster in clusters:
        indices = [i for i, value in enumerate(groups) if value == cluster]
        sums.append(int(difference[indices].sum()))
        sizes.append(int(difference[indices].size))
    sums, sizes = np.asarray(sums), np.asarray(sizes)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        chosen = rng.integers(len(clusters), size=len(clusters))
        differences.append(float(sums[chosen].sum() / sizes[chosen].sum()))
    return dict(
        baseline_accuracy=float((before.argmax(-1) == target).mean()),
        candidate_accuracy=float((after.argmax(-1) == target).mean()),
        difference=float(difference.mean()),
        interval95=np.quantile(differences, [0.025, 0.975], method="linear").tolist(),
        clusters=len(clusters),
        images=len(target),
        questions=int(target.size),
        replicates=replicates,
        seed=seed,
        method="paired whole-annotation-state cluster percentile; image/question-weighted ratio",
        blind_evaluation=False,
        observed_train_diagnostic=True,
    )


def attribution(native_logits, oracle_scores, predicted_scores, target):
    oracle_scores, predicted_scores, target = checked_scores(
        oracle_scores, predicted_scores, target
    )
    native_logits = np.asarray(native_logits)
    if native_logits.shape != (len(target), 4, 10) or not np.isfinite(native_logits).all():
        raise ValueError("aligned finite native image x field x ten-choice scores required")
    native = native_logits.argmax(-1) == target % 10
    oracle = oracle_scores.argmax(-1) == target
    predicted = predicted_scores.argmax(-1) == target
    counts = Counter(
        f"native_{int(n)}_oracle_{int(o)}_predicted_{int(p)}"
        for n, o, p in zip(native.flat, oracle.flat, predicted.flat, strict=True)
    )
    return dict(
        questions=int(target.size),
        native_fact_accuracy=float(native.mean()),
        oracle_decision_accuracy=float(oracle.mean()),
        predicted_decision_accuracy=float(predicted.mean()),
        intersections=dict(counts),
    )
