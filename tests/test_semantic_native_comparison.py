"""CPU tests for the fixed native comparison without model or GPU execution."""

import unittest

import numpy as np

from scripts.semantic_native_comparison import (
    attribution,
    compare_public_pairs,
    paired_accuracy_interval,
    permutation_check,
    replay_file_names,
)


class NativeComparisonTest(unittest.TestCase):
    def test_replay_rejects_missing_and_extra_payload(self):
        for actual in (["a.npy"], ["a.npy", "b.json", "extra.npy"]):
            with self.assertRaises(ValueError):
                replay_file_names(actual, {"a.npy": "hash1", "b.json": "hash2"})
        self.assertEqual(
            replay_file_names(["b.json", "a.npy"], {"a.npy": "1", "b.json": "2"}),
            ["a.npy", "b.json"],
        )

    def test_permutation_checks_actual_ids_on_ties(self):
        scores = np.zeros((1, 4, 40))
        permutation = np.arange(40)[::-1]
        with self.assertRaises(ValueError):
            permutation_check(scores, scores[:, :, permutation], permutation, 1e-5)

    def test_permutation_unique_winner_passes(self):
        scores = np.arange(160).reshape(1, 4, 40).astype(float)
        permutation = np.random.default_rng(17).permutation(40)
        result = permutation_check(scores, scores[:, :, permutation], permutation, 1e-5)
        self.assertTrue(result["predicted_option_ids_exact"])
        self.assertEqual(result["max_abs"], 0.0)

    def test_public_inputs_exclude_private_metadata(self):
        r = dict(
            state={"color": "red", "texture": "metal", "count": "2", "shape": "cube"},
            question="What is the color of the objects?",
            options=["color: red", "count: 2"],
            target=0,
            source="secret",
            field="DO NOT ROUTE",
            oracle={"private": "value"},
        )
        a = compare_public_pairs(r)
        r.update(target=1, source="changed", field="changed", oracle={})
        self.assertEqual(a, compare_public_pairs(r))
        self.assertEqual(
            a[0][0], "Question: What is the color of the objects?\nCandidate answer: color: red"
        )
        self.assertEqual(
            a[0][1], 'State: {"color":"red","count":"2","shape":"cube","texture":"metal"}'
        )

    def test_all_candidates_preserved_on_permutation(self):
        r = dict(state={}, question="Q", options=["A", "B", "C"])
        a = compare_public_pairs(r)
        r["options"] = ["C", "A", "B"]
        self.assertEqual(compare_public_pairs(r), [a[2], a[0], a[1]])

    def test_bootstrap_unequal_clusters_matches_direct_resampling(self):
        truth = np.array([[0, 10, 20, 30]] * 4)
        before = np.full((4, 4, 40), -1.0)
        after = before.copy()
        for i in range(4):
            for q in range(4):
                before[i, q, truth[i, q] if i == 0 else (truth[i, q] + 1)] = 1
                after[i, q, truth[i, q] if i in (1, 2) else (truth[i, q] + 1)] = 1
        # a has two images; its question count must contribute twice b/c.
        groups = ["a", "a", "b", "c"]
        actual = paired_accuracy_interval(before, after, truth, groups, replicates=101, seed=17)
        difference = (after.argmax(-1) == truth).astype(float) - (before.argmax(-1) == truth)
        rng = np.random.default_rng(17)
        values = []
        clusters = [np.array([0, 1]), np.array([2]), np.array([3])]
        for _ in range(101):
            chosen = np.concatenate([clusters[i] for i in rng.integers(3, size=3)])
            values.append(difference[chosen].mean())
        self.assertEqual(actual["difference"], 0.25)
        np.testing.assert_array_equal(
            actual["interval95"], np.quantile(values, [0.025, 0.975], method="linear")
        )
        self.assertEqual(actual["clusters"], 3)
        self.assertEqual(actual["images"], 4)

    def test_bootstrap_identical_scores(self):
        score = np.zeros((2, 4, 40))
        truth = np.array([[0, 10, 20, 30]] * 2)
        r = paired_accuracy_interval(score, score, truth, ["a", "b"], replicates=20)
        self.assertEqual(r["difference"], 0.0)
        self.assertEqual(r["interval95"], [0.0, 0.0])

    def test_bootstrap_rejects_invalid_inputs(self):
        score = np.zeros((2, 4, 40))
        truth = np.array([[0, 10, 20, 30]] * 2)
        for target, groups in [
            (truth, ["a"]),
            (truth.astype(float), ["a", "b"]),
            (truth + 100, ["a", "b"]),
        ]:
            with self.assertRaises(ValueError):
                paired_accuracy_interval(score, score, target, groups, replicates=20)
        score[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            paired_accuracy_interval(score, score, truth, ["a", "b"], replicates=20)

    def test_attribution_intersections(self):
        truth = np.array([[0, 10, 20, 30]] * 2)
        native = np.zeros((2, 4, 10))
        native[0, :, 0] = 1
        native[1, :, 1] = 1
        oracle = np.zeros((2, 4, 40))
        predicted = np.zeros_like(oracle)
        for i in range(2):
            for q in range(4):
                oracle[i, q, truth[i, q]] = 1
                predicted[i, q, truth[i, q] + i] = 1
        r = attribution(native, oracle, predicted, truth)
        self.assertEqual(r["questions"], 8)
        self.assertEqual(r["native_fact_accuracy"], 0.5)
        self.assertEqual(r["oracle_decision_accuracy"], 1.0)
        self.assertEqual(r["predicted_decision_accuracy"], 0.5)
        self.assertEqual(
            r["intersections"],
            {"native_1_oracle_1_predicted_1": 4, "native_0_oracle_1_predicted_0": 4},
        )


if __name__ == "__main__":
    unittest.main()
