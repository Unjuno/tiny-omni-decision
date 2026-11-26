"""Small numerical fixtures for the independent native result verifier."""

import math
import unittest

import numpy as np

from scripts import verify_semantic_native_comparison as VERIFY


class NativeVerifierTests(unittest.TestCase):
    def test_known_probability_scores(self):
        result = VERIFY.measure(np.log([[0.8, 0.2], [0.2, 0.8]]), np.array([0, 1]))
        for key, expected in dict(
            count=2, accuracy=1, nll=-math.log(0.8), brier=0.08, ece=0.2, mean_confidence=0.8
        ).items():
            self.assertAlmostEqual(result[key], expected)

    def test_uniform_forty_options(self):
        result = VERIFY.measure(np.zeros((4, 40)), np.array([0, 10, 20, 30]))
        self.assertAlmostEqual(result["accuracy"], 0.25)
        self.assertAlmostEqual(result["nll"], math.log(40))
        self.assertAlmostEqual(result["brier"], 0.975)
        self.assertAlmostEqual(result["ece"], 0.225)

    def test_empty_question_cannot_solve_changed_question_pair(self):
        result = VERIFY.summary(
            np.zeros((2, 4, 40)),
            np.array([[0, 10, 20, 30]] * 2),
            ["color", "texture", "count", "shape"],
        )
        self.assertEqual(result["changed_question_pair_both_correct"], 0)
        self.assertEqual(result["pair_count"], 12)
        self.assertEqual(result["field_routing_accuracy"], 0.25)
        self.assertEqual(result["within_correct_field_accuracy"], 1)
        self.assertEqual(result["all_four_correct"], 0)

    def test_score_shift_invariance(self):
        values = np.arange(80).reshape(2, 40).astype(float) / 10
        np.testing.assert_allclose(
            VERIFY.probabilities(values), VERIFY.probabilities(values + 1000), atol=1e-14, rtol=0
        )

    def test_nan_and_bad_labels_rejected(self):
        for scores, labels in [
            (np.array([[0, np.nan]]), np.array([0])),
            (np.zeros((1, 2)), np.array([2])),
            (np.zeros((1, 2)), np.array([0.0])),
        ]:
            with self.assertRaises(ValueError):
                VERIFY.measure(scores, labels)

    def test_nested_comparison_rejects_missing_field_or_changed_metric(self):
        for right in ({}, {"accuracy": 0.4}):
            with self.assertRaises(ValueError):
                VERIFY.equal({"accuracy": 0.5}, right)

    def test_whole_cluster_bootstrap_constant_effect(self):
        target = np.array([[0, 10, 20, 30]] * 3)
        before, after = np.zeros((3, 4, 40)), np.zeros((3, 4, 40))
        for i in range(3):
            for q in range(4):
                before[i, q, target[i, q] + 1] = 1
                after[i, q, target[i, q]] = 1
        result = VERIFY.bootstrap(before, after, target, ["a", "a", "b"])
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["difference"], 1)
        self.assertEqual(result["interval95"], [1, 1])


if __name__ == "__main__":
    unittest.main()
