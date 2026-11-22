from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.pretrained_reuse.train_spoken_palette_decision import metrics, tensor_batch
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.miniwob_dino_readout import permute_candidate_axis


class SpokenPaletteRunnerTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.arange(2 * 384, dtype=np.float32).reshape(2, 384)
        self.visual = np.arange(2 * 4 * 1154, dtype=np.float32).reshape(2, 4, 1154)
        self.arrays = {
            "audio_indices": np.asarray([0, 1]),
            "screen_indices": np.asarray([0, 1]),
            "option_positions": np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]]),
            "targets": np.asarray([1, 2]),
        }
        self.mean = np.zeros(1538, dtype=np.float32)
        self.std = np.ones(1538, dtype=np.float32)

    def test_tensor_batch_keeps_option_audio_and_visual_features_aligned(self):
        features, targets = tensor_batch(
            np.asarray([0, 1]), self.arrays, self.audio, self.visual, self.mean, self.std
        )
        self.assertEqual(tuple(features.shape), (2, 4, 1538))
        self.assertEqual(targets.tolist(), [1, 2])
        self.assertTrue(torch.equal(features[0, 0, :384], torch.from_numpy(self.audio[0])))
        self.assertTrue(torch.equal(features[0, 3, 384:], torch.from_numpy(self.visual[0, 3])))

    def test_unimodal_mean_replacements_preserve_only_intended_features(self):
        features, _ = tensor_batch(
            np.asarray([0]),
            self.arrays,
            self.audio,
            self.visual,
            self.mean,
            self.std,
            missing_audio=True,
        )
        self.assertTrue(torch.count_nonzero(features[:, :, :384]) == 0)
        self.assertGreater(torch.count_nonzero(features[:, :, 384:]), 0)

        features, _ = tensor_batch(
            np.asarray([0]),
            self.arrays,
            self.audio,
            self.visual,
            self.mean,
            self.std,
            missing_visual=True,
        )
        self.assertGreater(torch.count_nonzero(features[:, :, :384]), 0)
        self.assertTrue(torch.count_nonzero(features[:, :, 384:1536]) == 0)
        self.assertGreater(torch.count_nonzero(features[:, :, 1536:]), 0)

    def test_uniform_probabilities_have_four_way_chance_metrics(self):
        probabilities = np.full((4, 4), 0.25)
        targets = np.arange(4)
        colors = np.asarray(["blue", "green", "red", "white"], dtype=object)
        result = metrics(probabilities, targets, colors)
        self.assertAlmostEqual(result["accuracy"], 0.25)
        self.assertAlmostEqual(result["macro_color_accuracy"], 0.25)
        self.assertAlmostEqual(result["nll"], np.log(4))
        self.assertAlmostEqual(result["brier"], 0.75)
        self.assertAlmostEqual(result["ece_15"], 0.0)

    def test_shared_candidate_scorer_is_invariant_to_option_order(self):
        torch.manual_seed(17)
        scorer = FrozenFeatureCandidateScorer(1538, hidden_size=64).eval()
        features, _ = tensor_batch(
            np.asarray([0]), self.arrays, self.audio, self.visual, self.mean, self.std
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted, inverse = permute_candidate_axis(features, permutation)
        with torch.inference_mode():
            original = torch.softmax(scorer(features)[0], dim=-1)
            restored = torch.softmax(scorer(permuted)[0], dim=-1).index_select(0, inverse)
        self.assertLess(float((original - restored).abs().max()), 1e-6)

    def test_pinned_config_contains_no_duplicate_json_keys(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/pretrained_reuse/spoken_palette_decision_v1.json"
        )

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        config = json.loads(
            config_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
        self.assertEqual(config["training"]["selector_min_delta"], 0.0001)


if __name__ == "__main__":
    unittest.main()
