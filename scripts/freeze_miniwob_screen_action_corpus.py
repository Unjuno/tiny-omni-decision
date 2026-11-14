"""Freeze a small screen-only MiniWoB++ click-color train/dev corpus.

The script uses disjoint seed windows and never exposes utterances, DOM fields,
element names, or the browser's color labels in model-visible rows. The target
swatch must be checked in the live browser because MiniWoB's pruned returned
DOM can omit empty visual elements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from miniwob_screen_action_data import (
    resolve_screen_target_index,
    validate_screen_action_row,
    write_canonical_jsonl,
)
from PIL import Image

SOURCE_COMMIT = "33c3b4ddef8c6eb67c57a29663d844b1eda7e614"
TASK_ID = "miniwob/click-color-v1:screen-only"
SPLIT_WINDOWS = {
    "train": (940_000, 942_000),
    "validation": (950_000, 952_000),
}
OPTION_ORDER_XOR = 0x5A17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pixel_at(image: np.ndarray, rect: list[float]) -> tuple[int, int, int]:
    x, y, width, height = rect
    center_x = int(round(x + width / 2))
    center_y = int(round(y + height / 2))
    if not (0 <= center_x < image.shape[1] and 0 <= center_y < image.shape[0]):
        raise ValueError("rendered target or option lies outside the screenshot")
    return tuple(int(value) for value in image[center_y, center_x])


def _save_or_verify_screenshot(
    image: np.ndarray, image_path: Path, *, reuse_existing: bool = False
) -> str:
    if reuse_existing:
        if not image_path.is_file():
            raise FileNotFoundError(f"recovery screenshot is missing: {image_path}")
        with Image.open(image_path) as saved:
            saved_pixels = np.asarray(saved.convert("RGB"))
        if saved_pixels.shape != image.shape or not np.array_equal(saved_pixels, image):
            raise ValueError(f"replayed screenshot differs from preserved media: {image_path}")
    else:
        Image.fromarray(image).save(image_path, format="PNG", optimize=False)
    return _sha256(image_path)


def _configure_edge(args: argparse.Namespace) -> None:
    if not args.edge_compat:
        return
    if not args.browser_binary or not args.driver:
        raise ValueError("--edge-compat requires --browser-binary and --driver")
    os.environ["MINIWOB_CHROME_BINARY"] = str(args.browser_binary.resolve(strict=True))
    os.environ["MINIWOB_CHROMEDRIVER"] = str(args.driver.resolve(strict=True))
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService

    webdriver.Chrome = webdriver.Edge
    webdriver.ChromeOptions = EdgeOptions
    import miniwob.selenium_instance as selenium_instance

    selenium_instance.ChromeService = EdgeService


def _browser_state(driver: Any) -> dict[str, Any]:
    return driver.execute_script(
        """const query=document.querySelector('#query');
        const swatch=document.querySelector('#query-color');
        const rect=e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]};
        const targetColor=swatch
          ? (((swatch.getAttribute('style')||'').match(/background-color:\s*([^;]+)/i)||[])[1])
          : null;
        return {
          queryChildTag:query.children[0]?query.children[0].tagName:null,
          swatch:swatch?{visible:!!(swatch.offsetWidth||swatch.offsetHeight),rect:rect(swatch)}:null,
          targetColor:targetColor?targetColor.trim().toLowerCase():null,
          options:[...document.querySelectorAll('#area .color')].map(e=>({
            color:e.getAttribute('data-color'),rect:rect(e)
          }))
        };"""
    )


def _collect_split(
    *,
    env: Any,
    output_dir: Path,
    split: str,
    seed_range: tuple[int, int],
    count: int,
    seen_screen_hashes: set[str],
    reuse_existing_screenshots: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    start_seed, stop_seed = seed_range
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    used_seeds: list[int] = []
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=reuse_existing_screenshots)

    for seed in range(start_seed, stop_seed):
        observation, _ = env.reset(seed=seed)
        browser = _browser_state(env.unwrapped.instance.driver)
        swatch = browser["swatch"]
        if swatch is None or not swatch["visible"]:
            audit.append({"seed": seed, "accepted": False, "reason": "no_visible_target_swatch"})
            continue

        option_colors = [item["color"] for item in browser["options"]]
        if any(not isinstance(color, str) for color in option_colors):
            raise ValueError(f"missing rendered option label in seed {seed}")
        target_name = browser["targetColor"]
        if not target_name:
            raise ValueError(f"missing target color in rendered swatch at seed {seed}")
        image = observation["screenshot"]
        swatch_rgb = _pixel_at(image, swatch["rect"])
        option_rgbs = [_pixel_at(image, item["rect"]) for item in browser["options"]]
        target_source_index = resolve_screen_target_index(
            query_child_tag=browser["queryChildTag"],
            query_swatch_visible=bool(swatch["visible"]),
            target_color=target_name,
            option_colors=option_colors,
            swatch_rgb=swatch_rgb,
            option_center_rgbs=option_rgbs,
        )
        if len(browser["options"]) != 4:
            raise ValueError(f"expected four action choices at seed {seed}")
        image_bytes = np.ascontiguousarray(image).tobytes()
        pixel_hash = hashlib.sha256(image_bytes).hexdigest()
        if pixel_hash in seen_screen_hashes:
            audit.append({"seed": seed, "accepted": False, "reason": "duplicate_screen_pixels"})
            continue
        seen_screen_hashes.add(pixel_hash)

        height, width = image.shape[:2]
        order_seed = seed ^ OPTION_ORDER_XOR
        permutation = np.random.default_rng(order_seed).permutation(4).tolist()
        target_index = permutation.index(target_source_index)
        options = []
        for output_index, source_index in enumerate(permutation):
            x, y, box_width, box_height = browser["options"][source_index]["rect"]
            options.append(
                {
                    "option_id": f"action-{output_index}",
                    "x_norm": (x + box_width / 2) / width,
                    "y_norm": (y + box_height / 2) / height,
                }
            )

        sample_id = f"{split}-seed-{seed}"
        image_path = split_dir / f"{sample_id}.png"
        screenshot_sha256 = _save_or_verify_screenshot(
            image, image_path, reuse_existing=reuse_existing_screenshots
        )
        row = {
            "sample_id": sample_id,
            "episode_group_id": sample_id,
            "task_template": TASK_ID,
            "seed": seed,
            "screenshot_path": f"{split}/{image_path.name}",
            "screenshot_sha256": screenshot_sha256,
            "screenshot_shape": list(image.shape),
            "option_order_seed": order_seed,
            "options": options,
            "target_index": target_index,
        }
        validate_screen_action_row(row)
        rows.append(row)
        used_seeds.append(seed)
        audit.append(
            {
                "seed": seed,
                "accepted": True,
                "target_color": target_name,
                "target_rgb": list(swatch_rgb),
                "option_colors": option_colors,
                "option_center_rgbs": [list(value) for value in option_rgbs],
                "target_source_index": target_source_index,
                "model_row_target_index": target_index,
                "pixel_match_verified": True,
                "model_inputs_exclude_text_dom_and_color_names": True,
                "screenshot_pixel_sha256": pixel_hash,
            }
        )
        if len(rows) == count:
            break

    if len(rows) != count:
        raise RuntimeError(
            f"{split} reached only {len(rows)}/{count} unique valid rows in seed window "
            f"[{start_seed}, {stop_seed})"
        )
    return rows, audit, used_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-split", type=int, default=512)
    parser.add_argument("--edge-compat", action="store_true")
    parser.add_argument("--browser-binary", type=Path)
    parser.add_argument("--driver", type=Path)
    args = parser.parse_args()
    if args.count_per_split < 2:
        raise ValueError("each split must contain at least two examples")
    source_root = args.source_root.resolve(strict=True)
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"MiniWoB++ source must be pinned at {SOURCE_COMMIT}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite corpus output: {output_dir}")
    output_dir.mkdir(parents=True)
    _configure_edge(args)
    sys.path.insert(0, str(source_root))
    import gymnasium
    import miniwob

    gymnasium.register_envs(miniwob)
    env = gymnasium.make("miniwob/click-color-v1", render_mode=None, wait_ms=200)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_audit: dict[str, list[dict[str, Any]]] = {}
    all_seeds: dict[str, list[int]] = {}
    seen_hashes: set[str] = set()
    try:
        for split, seed_range in SPLIT_WINDOWS.items():
            rows, audit, seeds = _collect_split(
                env=env,
                output_dir=output_dir,
                split=split,
                seed_range=seed_range,
                count=args.count_per_split,
                seen_screen_hashes=seen_hashes,
            )
            all_rows[split], all_audit[split], all_seeds[split] = rows, audit, seeds
    finally:
        env.close()

    if set(all_seeds["train"]) & set(all_seeds["validation"]):
        raise ValueError("train and validation seeds overlap")
    hashes = {
        split: write_canonical_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in all_rows.items()
    }
    audit_hashes = {
        split: write_canonical_jsonl(output_dir / f"collector-audit-{split}.jsonl", rows)
        for split, rows in all_audit.items()
    }
    manifest = {
        "status": "FROZEN_DEVELOPMENT_CORPUS_NO_MODEL_EVALUATION",
        "source_repository": "https://github.com/Farama-Foundation/miniwob-plusplus",
        "source_commit": source_commit,
        "source_license": "MIT",
        "task": TASK_ID,
        "count_per_split": args.count_per_split,
        "splits": {
            split: {
                "seed_window_half_open": list(SPLIT_WINDOWS[split]),
                "seed_order_sha256": hashlib.sha256(
                    ",".join(map(str, all_seeds[split])).encode()
                ).hexdigest(),
                "examples_sha256": hashes[split],
                "collector_audit_sha256": audit_hashes[split],
                "example_count": len(all_rows[split]),
                "unique_episode_groups": len({row["episode_group_id"] for row in all_rows[split]}),
                "unique_screenshots": len({row["screenshot_sha256"] for row in all_rows[split]}),
            }
            for split in all_rows
        },
        "cross_split_seed_overlap": 0,
        "cross_split_screenshot_pixel_hash_overlap": 0,
        "model_inputs": ["screenshot RGB", "opaque action IDs", "normalized action coordinates"],
        "model_inputs_exclude": [
            "utterance text",
            "DOM",
            "browser target field",
            "element labels",
            "CSS color names",
        ],
        "collector_audit_is_not_model_input": True,
        "controls_to_freeze_before_scoring": [
            "candidate-position prior",
            "zeroed screenshot",
            "permuted screenshot across episodes",
            "same-example candidate-order permutation",
        ],
        "validation_is_development_only": True,
        "sealed_audit_created": False,
        "quality_metrics_reported": False,
        "runtime_note": "Windows/Edge compatibility smoke only; upstream supports Linux and macOS.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
