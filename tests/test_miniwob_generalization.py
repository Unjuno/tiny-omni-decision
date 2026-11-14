from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from freeze_miniwob_screen_action_corpus import _save_or_verify_screenshot
from PIL import Image

from tiny_omni_decision.miniwob_generalization import (
    assert_seed_window_fresh,
    episode_bootstrap,
    metric_arrays,
    paired_bootstrap_difference,
    pixel_sha256,
    previous_usage,
    sha256_file,
    target_colors_by_sample_id,
    validate_new_rows,
)


def _row(root: Path, sample_id: str, seed: int, image: Image.Image) -> dict:
    path = root / f"{sample_id}.png"
    image.save(path)
    return {
        "sample_id": sample_id,
        "episode_group_id": sample_id,
        "task_template": "miniwob/click-color-v1:screen-only",
        "seed": seed,
        "screenshot_path": path.name,
        "screenshot_sha256": sha256_file(path),
        "screenshot_shape": [image.height, image.width, 3],
        "option_order_seed": seed ^ 0x5A17,
        "options": [
            {"option_id": "action-0", "x_norm": 0.25, "y_norm": 0.5},
            {"option_id": "action-1", "x_norm": 0.75, "y_norm": 0.5},
        ],
        "target_index": 0,
    }


class MiniWoBGeneralizationTests(unittest.TestCase):
    def test_recovery_reuses_only_identical_preserved_screenshots(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "screen.png"
            pixels = np.zeros((3, 4, 3), dtype=np.uint8)
            pixels[:, :, 1] = 127
            first_hash = _save_or_verify_screenshot(pixels, image_path)

            replay_hash = _save_or_verify_screenshot(pixels.copy(), image_path, reuse_existing=True)

            self.assertEqual(first_hash, replay_hash)
            changed = pixels.copy()
            changed[0, 0] = [1, 2, 3]
            with self.assertRaisesRegex(ValueError, "differs from preserved media"):
                _save_or_verify_screenshot(changed, image_path, reuse_existing=True)

    def test_previous_usage_reads_seed_windows_jsonl_and_decoded_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "history"
            root.mkdir()
            image = Image.new("RGB", (4, 3), (20, 30, 40))
            image.save(root / "screen-seed-12.png")
            (root / "manifest.json").write_text(
                json.dumps({"splits": {"train": {"seed_window_half_open": [10, 13]}}}),
                encoding="utf-8",
            )
            (root / "examples.jsonl").write_text(
                json.dumps({"sample_id": "train-seed-12", "seed": 12}) + "\n",
                encoding="utf-8",
            )

            inventory = previous_usage([root])

            self.assertEqual(inventory["seed_values"], [10, 11, 12])
            self.assertEqual(inventory["screenshot_pixel_count"], 1)
            self.assertIn(pixel_sha256(image), inventory["pixel_hashes"])

    def test_seed_window_rejects_historical_seed_overlap(self):
        assert_seed_window_fresh([100, 200], [99, 200, 201])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            assert_seed_window_fresh([100, 200], [150])

    def test_previous_usage_tracks_masked_screen_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "history"
            root.mkdir()
            first = Image.new("RGB", (160, 210), "white")
            second = first.copy()
            first.paste((255, 0, 0), (0, 0, 160, 64))
            second.paste((0, 0, 255), (0, 0, 160, 64))
            first.save(root / "screen-seed-1.png")
            second.save(root / "screen-seed-2.png")

            inventory = previous_usage([root])

            self.assertEqual(inventory["screenshot_pixel_count"], 2)
            self.assertEqual(inventory["masked_pixel_count"], 1)
            self.assertEqual(len(inventory["masked_pixel_hashes"]), 1)

    def test_collector_audit_color_labels_join_by_seed(self):
        rows = [
            {"sample_id": "evaluation-seed-7", "seed": 7},
            {"sample_id": "evaluation-seed-8", "seed": 8},
        ]
        audit = [
            {"seed": 7, "accepted": True, "target_color": "red"},
            {"seed": 8, "accepted": False, "reason": "excluded"},
            {"seed": 8, "accepted": True, "target_color": "blue"},
        ]
        self.assertEqual(
            target_colors_by_sample_id(rows, audit),
            {"evaluation-seed-7": "red", "evaluation-seed-8": "blue"},
        )
        with self.assertRaisesRegex(ValueError, "count differs"):
            target_colors_by_sample_id(rows, audit[:1])

    def test_new_corpus_validation_requires_unique_seeds_groups_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            corpus.mkdir()
            image_a = Image.new("RGB", (4, 3), (1, 2, 3))
            image_b = Image.new("RGB", (4, 3), (4, 5, 6))
            rows = [
                _row(corpus, "evaluation-seed-100", 100, image_a),
                _row(corpus, "evaluation-seed-101", 101, image_b),
            ]

            receipt = validate_new_rows(
                rows,
                corpus_root=corpus,
                seed_window=[100, 102],
                expected_count=2,
                previous_seeds=[10],
                previous_pixel_hashes=[],
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["unique_episode_groups"], 2)
            self.assertEqual(receipt["previous_screenshot_pixel_overlap"], 0)

            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                validate_new_rows(
                    [rows[0], {**rows[1], "seed": 100}],
                    corpus_root=corpus,
                    seed_window=[100, 102],
                    expected_count=2,
                    previous_seeds=[],
                    previous_pixel_hashes=[],
                )

            with self.assertRaisesRegex(ValueError, "overlap a previous"):
                validate_new_rows(
                    rows,
                    corpus_root=corpus,
                    seed_window=[100, 102],
                    expected_count=2,
                    previous_seeds=[],
                    previous_pixel_hashes=[pixel_sha256(image_b)],
                )

    def test_metrics_and_episode_bootstrap_are_deterministic_and_paired(self):
        probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
        targets = np.asarray([0, 1, 1], dtype=np.int64)
        metrics = metric_arrays(probabilities, targets)
        self.assertEqual(metrics["count"], 3)
        self.assertTrue(math.isclose(metrics["accuracy"], 2 / 3))
        self.assertGreater(metrics["nll"], 0)
        first = episode_bootstrap(probabilities, targets, repetitions=200, seed=17)
        second = episode_bootstrap(probabilities, targets, repetitions=200, seed=17)
        self.assertEqual(first, second)
        same = paired_bootstrap_difference(
            probabilities, probabilities, targets, repetitions=200, seed=1701
        )
        self.assertEqual(same["accuracy_a_minus_b_95_percentile_ci"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
