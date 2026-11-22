from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.spoken_palette_decision_data import (
    COLORS,
    build_rows_for_split,
    factorial_support,
    layouts_by_parity,
    split_speaker_groups,
)


def _rows_for_groups(group_count: int = 4) -> list[dict]:
    rows = []
    for group_index in range(group_count):
        group = f"speaker-{group_index}"
        for color in COLORS:
            source_id = f"{group}-{color}"
            rows.append(
                {
                    "source_id": source_id,
                    "speaker": group,
                    "speaker_group": group,
                    "targets": {"color": color},
                }
            )
    return rows


class SpokenPaletteDecisionDataTests(unittest.TestCase):
    def test_parity_partition_has_balanced_color_position_marginals(self):
        groups = layouts_by_parity()
        self.assertEqual(
            {parity: len(layouts) for parity, layouts in groups.items()},
            {0: 12, 1: 12},
        )
        for layouts in groups.values():
            for color in COLORS:
                counts = [sum(layout[pos] == color for layout in layouts) for pos in range(4)]
                self.assertEqual(counts, [3] * 4)

    def test_speaker_split_is_deterministic_and_indivisible(self):
        rows = _rows_for_groups()
        first = split_speaker_groups(rows, counts=(1, 1, 1, 1))
        second = split_speaker_groups(rows, counts=(1, 1, 1, 1))
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "selection", "calibration", "evaluation"})
        for row in rows:
            self.assertEqual(first[row["speaker_group"]], first[row["speaker"]])

    def test_generated_factorial_pairs_require_both_audio_and_screen(self):
        rows = _rows_for_groups()
        assignment = split_speaker_groups(rows, counts=(1, 1, 1, 1))
        source_index = {row["source_id"]: index for index, row in enumerate(rows)}
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            generated = {}
            for split in ("train", "selection", "calibration", "evaluation"):
                generated[split] = build_rows_for_split(
                    rows,
                    split=split,
                    speaker_splits=assignment,
                    audio_feature_index=source_index,
                    output_root=output_root,
                    variants_per_layout=2,
                )

            for examples in generated.values():
                support = factorial_support(examples, rows)
                self.assertTrue(support["audio_only_exactly_uninformative"])
                self.assertTrue(support["image_only_no_instance_signal_beyond_color_prior"])
                self.assertEqual(support["image_only_macro_accuracy_ceiling"], 0.25)
                self.assertTrue(support["both_inputs_determine_unique_action"])
                self.assertEqual(support["unique_audio_sources"], 4)
                self.assertEqual(support["rows"], 4 * 12 * 2)
                self.assertEqual(len({row["screenshot_sha256"] for row in examples}), 24)
                self.assertTrue(all("target_color" not in row for row in examples))
                self.assertTrue(
                    all("transcript" not in row and "utterance" not in row for row in examples)
                )
                for row in examples:
                    self.assertEqual(len(row["options"]), 4)
                    self.assertEqual(len({option["option_id"] for option in row["options"]}), 4)

            image_hashes = {
                split: {row["screenshot_sha256"] for row in examples}
                for split, examples in generated.items()
            }
            for left_index, left in enumerate(image_hashes):
                for right in tuple(image_hashes)[left_index + 1 :]:
                    self.assertFalse(image_hashes[left] & image_hashes[right])

    def test_split_fails_if_a_speaker_group_lacks_one_color(self):
        rows = _rows_for_groups()
        rows.pop()
        with self.assertRaisesRegex(ValueError, "all four color labels"):
            split_speaker_groups(rows, counts=(1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
