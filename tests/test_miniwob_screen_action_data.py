from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.miniwob_screen_action_data import (
    model_option_positions,
    output_option_ids,
    resolve_screen_target_index,
    validate_screen_action_row,
    write_canonical_jsonl,
)


class MiniWoBScreenActionDataTests(unittest.TestCase):
    def test_live_swatch_check_does_not_depend_on_pruned_observation_dom(self):
        # The MiniWoB observation's dom_elements can omit this empty DIV even
        # while it is visible. Eligibility therefore comes from browser JS.
        returned_dom_elements = [{"tag": "div", "id": "", "classes": "color"}]
        self.assertFalse(
            any(element.get("id") == "query-color" for element in returned_dom_elements)
        )
        self.assertEqual(
            resolve_screen_target_index(
                query_child_tag="DIV",
                query_swatch_visible=True,
                target_color="blue",
                option_colors=("red", "blue", "lime", "white"),
                swatch_rgb=(0, 0, 255),
                option_center_rgbs=((255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)),
            ),
            1,
        )

    def test_text_target_variant_is_not_admitted_as_screen_only(self):
        with self.assertRaisesRegex(ValueError, "visible target-color swatch"):
            resolve_screen_target_index(
                query_child_tag="SPAN",
                query_swatch_visible=False,
                target_color="blue",
                option_colors=("red", "blue", "lime", "white"),
                swatch_rgb=(0, 0, 255),
                option_center_rgbs=((255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)),
            )

    def test_rendered_pixels_must_uniquely_support_upstream_label(self):
        invalid_cases = [
            ((0, 0, 255), ((255, 0, 0), (0, 255, 0), (0, 255, 0), (255, 255, 255))),
            ((0, 0, 254), ((255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255))),
        ]
        for target_rgb, option_rgbs in invalid_cases:
            with self.subTest(target_rgb=target_rgb):
                with self.assertRaisesRegex(ValueError, "match only the labeled correct option"):
                    resolve_screen_target_index(
                        query_child_tag="DIV",
                        query_swatch_visible=True,
                        target_color="blue",
                        option_colors=("red", "blue", "lime", "white"),
                        swatch_rgb=target_rgb,
                        option_center_rgbs=option_rgbs,
                    )

    def test_screen_action_row_contains_no_utterance_dom_or_label_text(self):
        row = {
            "sample_id": "episode-1",
            "episode_group_id": "episode-1",
            "task_template": "miniwob/click-color-v1:screen-only",
            "seed": 11,
            "screenshot_path": "episode-1.png",
            "screenshot_sha256": "a" * 64,
            "screenshot_shape": [210, 160, 3],
            "option_order_seed": 17,
            "options": [
                {"option_id": "action-0", "x_norm": 0.25, "y_norm": 0.5},
                {"option_id": "action-1", "x_norm": 0.75, "y_norm": 0.5},
            ],
            "target_index": 1,
        }
        validate_screen_action_row(row)
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            validate_screen_action_row({**row, "utterance": "Click on blue"})
        self.assertEqual(model_option_positions(row), ((0.25, 0.5), (0.75, 0.5)))
        self.assertEqual(output_option_ids(row), ("action-0", "action-1"))
        self.assertNotIn(row["target_index"], model_option_positions(row))

    def test_jsonl_manifest_hash_matches_bytes_written_on_windows(self):
        rows = [{"sample_id": "one", "value": 1}, {"sample_id": "two", "value": 2}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            expected = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ).encode("utf-8")
            reported = write_canonical_jsonl(path, rows)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(reported, hashlib.sha256(path.read_bytes()).hexdigest())
