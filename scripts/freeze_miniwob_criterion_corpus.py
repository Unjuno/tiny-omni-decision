"""Freeze a fresh group-split MiniWoB criterion-conditioned corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from tiny_omni_decision.miniwob_criterion_corpus import (
    DEFAULT_MASK_ROWS,
    build_group_rows,
    criterion_text,
    masked_screen,
    pixel_sha256,
    validate_mask_layout,
    validate_split_groups,
)
from tiny_omni_decision.miniwob_generalization import previous_usage

SOURCE_REPOSITORY = "https://github.com/Farama-Foundation/miniwob-plusplus"
SOURCE_COMMIT = "33c3b4ddef8c6eb67c57a29663d844b1eda7e614"
OPTION_ORDER_XOR = 0x5A17


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect_split(
    *,
    env: Any,
    output_root: Path,
    split: str,
    seed_window: list[int],
    expected_groups: int,
    mask_rows: int,
    historical_full_hashes: set[str],
    historical_masked_hashes: set[str],
    seen_full_hashes: set[str],
    seen_masked_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[str]]:
    from freeze_miniwob_screen_action_corpus import (
        OPTION_ORDER_XOR as SOURCE_ORDER_XOR,
    )
    from freeze_miniwob_screen_action_corpus import (
        _browser_state,
        _pixel_at,
    )
    from miniwob_screen_action_data import resolve_screen_target_index

    if SOURCE_ORDER_XOR != OPTION_ORDER_XOR:
        raise ValueError("source screen-action option-order policy changed")
    image_dir = output_root / split / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    used_seeds: list[int] = []
    used_masked_hashes: list[str] = []
    for seed in range(seed_window[0], seed_window[1]):
        observation, _ = env.reset(seed=seed)
        browser = _browser_state(env.unwrapped.instance.driver)
        swatch = browser.get("swatch")
        options = browser.get("options", [])
        if swatch is None or not swatch.get("visible") or len(options) != 4:
            audit.append({"seed": seed, "accepted": False, "reason": "invalid_source_layout"})
            continue
        colors = [item.get("color") for item in options]
        if any(not isinstance(color, str) for color in colors):
            audit.append({"seed": seed, "accepted": False, "reason": "missing_option_color"})
            continue
        if len({" ".join(color.strip().lower().split()) for color in colors}) != 4:
            audit.append({"seed": seed, "accepted": False, "reason": "nonunique_option_colors"})
            continue
        try:
            for color in colors:
                criterion_text(color)
        except ValueError as error:
            audit.append({"seed": seed, "accepted": False,
                          "reason": "unsupported_criterion_color", "detail": str(error)})
            continue
        try:
            validate_mask_layout(
                query_rect=browser.get("queryRect"),
                swatch_rect=swatch["rect"],
                option_rects=[item["rect"] for item in options],
                mask_rows=mask_rows,
            )
        except (TypeError, ValueError) as error:
            audit.append({"seed": seed, "accepted": False,
                          "reason": "unsafe_instruction_mask_geometry", "detail": str(error)})
            continue
        source_target_color = browser.get("targetColor")
        if not isinstance(source_target_color, str) or not source_target_color:
            audit.append({"seed": seed, "accepted": False, "reason": "missing_original_target"})
            continue

        pixels = np.asarray(observation["screenshot"], dtype=np.uint8)
        if pixels.shape != (210, 160, 3):
            raise ValueError(f"unexpected source screenshot shape: {pixels.shape}")
        swatch_rgb = _pixel_at(pixels, swatch["rect"])
        option_rgbs = [_pixel_at(pixels, item["rect"]) for item in options]
        try:
            source_target_index = resolve_screen_target_index(
                query_child_tag=browser.get("queryChildTag"),
                query_swatch_visible=True,
                target_color=source_target_color.strip().lower(),
                option_colors=colors,
                swatch_rgb=swatch_rgb,
                option_center_rgbs=option_rgbs,
            )
        except ValueError as error:
            audit.append({"seed": seed, "accepted": False,
                          "reason": "source_target_pixel_check_failed", "detail": str(error)})
            continue

        full_hash = _sha256_bytes(np.ascontiguousarray(pixels).tobytes())
        if full_hash in historical_full_hashes or full_hash in seen_full_hashes:
            audit.append({"seed": seed, "accepted": False, "reason": "duplicate_full_screen"})
            continue
        masked = masked_screen(Image.fromarray(pixels), mask_rows=mask_rows)
        masked_hash = pixel_sha256(masked)
        if masked_hash in historical_masked_hashes or masked_hash in seen_masked_hashes:
            audit.append({"seed": seed, "accepted": False,
                          "reason": "duplicate_criterion_visible_screen"})
            continue

        positions = []
        for item in options:
            x, y, width, height = item["rect"]
            positions.append(((x + width / 2) / pixels.shape[1],
                              (y + height / 2) / pixels.shape[0]))
        group_id = f"miniwob-seed-{seed}"
        relative_image = Path(split) / "images" / f"{group_id}.png"
        image_path = output_root / relative_image
        masked.save(image_path, format="PNG", optimize=False)
        image_file_hash = _sha256_file(image_path)
        group_rows = build_group_rows(
            split=split,
            seed=seed,
            screenshot_path=relative_image.as_posix(),
            screenshot_sha256=image_file_hash,
            screenshot_pixel_sha256=masked_hash,
            option_colors=colors,
            option_positions=positions,
        )
        rows.extend(group_rows)
        criterion_targets = {
            row["question"]: row["target_option_id"] for row in group_rows
        }
        audit.append({
            "seed": seed,
            "accepted": True,
            "group_id": group_id,
            "source_target_color": source_target_color,
            "source_target_index_verified": source_target_index,
            "source_option_colors": colors,
            "source_option_center_rgbs": [list(rgb) for rgb in option_rgbs],
            "criterion_target_option_ids": criterion_targets,
            "original_instruction_and_swatch_masked": True,
            "mask_rows": mask_rows,
            "full_screen_pixel_sha256": full_hash,
            "masked_screen_pixel_sha256": masked_hash,
            "masked_screenshot_file_sha256": image_file_hash,
        })
        seen_full_hashes.add(full_hash)
        seen_masked_hashes.add(masked_hash)
        used_seeds.append(seed)
        used_masked_hashes.append(masked_hash)
        if len(used_seeds) == expected_groups:
            break

    if len(used_seeds) != expected_groups:
        raise RuntimeError(
            f"{split} reached {len(used_seeds)}/{expected_groups} unique groups in "
            f"fixed seed window {seed_window}; do not widen the window implicitly"
        )
    return rows, audit, used_seeds, used_masked_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--browser-binary", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sealed-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("option_order_seed_xor") != OPTION_ORDER_XOR:
        raise ValueError("config option order does not match the tested policy")
    output_root = args.output_dir.resolve()
    sealed_root = args.sealed_dir.resolve()
    if output_root.exists() or sealed_root.exists():
        raise FileExistsError("refusing to overwrite a corpus or sealed-audit output")
    nested_output = output_root in sealed_root.parents or sealed_root in output_root.parents
    if output_root == sealed_root or nested_output:
        raise ValueError("sealed audit must use a separate non-nested output directory")

    source_root = args.source_root.resolve(strict=True)
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT or config["source"]["commit"] != SOURCE_COMMIT:
        raise ValueError("MiniWoB++ source does not match the pinned commit")
    license_path = source_root / "LICENSE"
    license_hash = _sha256_file(license_path)
    if license_hash != config["source"]["license_sha256"]:
        raise ValueError("MiniWoB++ license receipt differs from the pinned hash")

    roots = [Path(value).resolve(strict=True) for value in config["historical_artifacts"]]
    prior = previous_usage(roots, config.get("historical_seeds_without_saved_corpus", []))
    historical_full_hashes = set(prior["pixel_hashes"])
    historical_full_hashes.update(
        config.get("historical_full_pixel_hashes_without_saved_screenshots", [])
    )
    historical_masked_hashes = set(prior["masked_pixel_hashes"])
    historical_masked_hashes.update(
        config.get("historical_masked_pixel_hashes_without_saved_screenshots", [])
    )
    windows = config["seed_windows"]
    mask_rows = config.get("mask_rows", DEFAULT_MASK_ROWS)
    if mask_rows != DEFAULT_MASK_ROWS:
        raise ValueError(f"mask_rows is pinned to the tested value {DEFAULT_MASK_ROWS}")
    split_names = ("train", "validation", "sealed_audit")
    all_seeds = prior["seed_values"]
    all_windows = [windows[name] for name in split_names]
    flat_seed_windows = [(name, pair) for name, pair in zip(split_names, all_windows, strict=True)]
    for name, window in flat_seed_windows:
        if (len(window) != 2 or type(window[0]) is not int or type(window[1]) is not int
                or window[0] < 0 or window[1] <= window[0]):
            raise ValueError(f"invalid fixed half-open seed window for {name}")
        overlap = [seed for seed in all_seeds if window[0] <= seed < window[1]]
        if overlap:
            raise ValueError(f"{name} seed window overlaps {len(overlap)} historical seeds")
    for i, (name_a, a) in enumerate(flat_seed_windows):
        for name_b, b in flat_seed_windows[i + 1:]:
            if max(a[0], b[0]) < min(a[1], b[1]):
                raise ValueError(f"seed windows overlap: {name_a} and {name_b}")

    edge_args = SimpleNamespace(edge_compat=True, browser_binary=args.browser_binary,
                                driver=args.driver)
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from freeze_miniwob_screen_action_corpus import _configure_edge
    from miniwob_screen_action_data import write_canonical_jsonl

    _configure_edge(edge_args)
    output_root.mkdir(parents=True)
    sealed_root.mkdir(parents=True)
    effective_config = dict(config)
    effective_config["execution"] = {
        "source_root": str(source_root),
        "browser_binary": str(args.browser_binary.resolve(strict=True)),
        "driver": str(args.driver.resolve(strict=True)),
        "driver_sha256": _sha256_file(args.driver.resolve(strict=True)),
        "output_dir": str(output_root),
        "sealed_audit_dir": str(sealed_root),
    }
    effective_config_path = output_root / "effective-config.json"
    _write_json(effective_config_path, effective_config)
    started = datetime.now(UTC)
    preflight = {
        "status": "PASS_BEFORE_GENERATION",
        "experiment_id": config["experiment_id"],
        "source_config_sha256": _sha256_file(config_path),
        "effective_config_sha256": _sha256_file(effective_config_path),
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": source_commit,
        "source_license_sha256": license_hash,
        "seed_windows": windows,
        "historical_usage": {
            "roots": prior["roots"],
            "seed_count": prior["seed_count"],
            "max_seed": max(all_seeds),
            "seed_values_sha256": prior["seed_values_sha256"],
            "full_screen_pixel_count": prior["screenshot_pixel_count"],
            "full_screen_pixel_sha256": prior["screenshot_pixel_sha256"],
            "masked_pixel_count": prior["masked_pixel_count"],
            "masked_pixel_sha256": prior["masked_pixel_sha256"],
            "explicit_full_pixel_hash_count": len(
                config.get("historical_full_pixel_hashes_without_saved_screenshots", [])
            ),
            "explicit_masked_pixel_hash_count": len(
                config.get("historical_masked_pixel_hashes_without_saved_screenshots", [])
            ),
        },
        "created_at_utc": started.isoformat(),
        "sealed_audit_path_not_exposed_to_trainer": True,
    }
    _write_json(output_root / "pre-generation-preflight.json", preflight)

    start_clock = time.perf_counter()
    seen_full = set(historical_full_hashes)
    seen_masked = set(historical_masked_hashes)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_audits: dict[str, list[dict[str, Any]]] = {}
    split_receipts: dict[str, dict[str, Any]] = {}
    roots_by_split = {"train": output_root, "validation": output_root,
                      "sealed_audit": sealed_root}
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    env = None
    try:
        import miniwob
        from gymnasium import make, register_envs

        register_envs(miniwob)
        env = make("miniwob/click-color-v1", render_mode=None, wait_ms=200)
        for split in split_names:
            print(f"collecting split={split} target_groups={config['group_counts'][split]}",
                  flush=True)
            rows, audit, seeds, mask_hashes = _collect_split(
                env=env,
                output_root=roots_by_split[split],
                split=split,
                seed_window=windows[split],
                expected_groups=config["group_counts"][split],
                mask_rows=mask_rows,
                historical_full_hashes=historical_full_hashes,
                historical_masked_hashes=historical_masked_hashes,
                seen_full_hashes=seen_full,
                seen_masked_hashes=seen_masked,
            )
            split_rows[split] = rows
            split_audits[split] = audit
            split_receipts[split] = {
                "seed_window_half_open": windows[split],
                "accepted_groups": len(seeds),
                "question_rows": len(rows),
                "unique_masked_screen_hashes": len(set(mask_hashes)),
                "seed_order_sha256": _sha256_bytes(",".join(map(str, seeds)).encode()),
            }
            print(f"collected split={split} groups={len(seeds)} questions={len(rows)}",
                  flush=True)
        validate_split_groups(split_rows)
        for split in ("train", "validation"):
            write_canonical_jsonl(output_root / f"{split}.jsonl", split_rows[split])
            write_canonical_jsonl(output_root / "audit" / f"collector-{split}.jsonl",
                                  split_audits[split])
        sealed_rows_hash = write_canonical_jsonl(
            sealed_root / "sealed_audit.jsonl", split_rows["sealed_audit"]
        )
        sealed_audit_hash = write_canonical_jsonl(
            sealed_root / "collector-audit.jsonl", split_audits["sealed_audit"]
        )
        sealed_manifest = {
            "status": "SEALED_AUDIT_FROZEN_NOT_OPENED",
            "experiment_id": config["experiment_id"],
            "source_commit": source_commit,
            "split": "sealed_audit",
            **split_receipts["sealed_audit"],
            "examples_sha256": sealed_rows_hash,
            "collector_audit_sha256": sealed_audit_hash,
            "model_loaded": False,
            "model_scored": False,
        }
        _write_json(sealed_root / "manifest.json", sealed_manifest)
        manifest = {
            "status": "TRAIN_VALIDATION_FROZEN_SEALED_AUDIT_ISOLATED",
            "experiment_id": config["experiment_id"],
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": source_commit,
            "source_license": "MIT",
            "source_license_sha256": license_hash,
            "source_config_sha256": _sha256_file(config_path),
            "effective_config_sha256": _sha256_file(effective_config_path),
            "task": config["task"],
            "task_contract": {
                "groups_per_screen": 4,
                "criteria": "one Select-the-color-button question per displayed option color",
                "model_visible_fields": ["masked screenshot RGB", "question text",
                                         "opaque option IDs", "normalized option coordinates"],
                "model_visible_exclusions": ["original instruction", "original target swatch",
                                             "DOM", "CSS color names", "source target label"],
                "sealed_audit_loaded_during_collection": False,
            },
            "mask_rows": mask_rows,
            "option_order_seed_xor": OPTION_ORDER_XOR,
            "historical_usage": preflight["historical_usage"],
            "cross_split_seed_overlap": 0,
            "cross_split_masked_screen_pixel_overlap": 0,
            "splits": {
                split: {
                    **split_receipts[split],
                    "examples_sha256": _sha256_file(
                        output_root / f"{split}.jsonl"
                    ),
                    "collector_audit_sha256": _sha256_file(
                        output_root / "audit" / f"collector-{split}.jsonl"
                    ),
                }
                for split in ("train", "validation")
            },
            "sealed_audit": {
                "directory": str(sealed_root),
                "manifest_sha256": _sha256_file(sealed_root / "manifest.json"),
                "examples_sha256": sealed_rows_hash,
                "question_rows": len(split_rows["sealed_audit"]),
                "groups": split_receipts["sealed_audit"]["accepted_groups"],
            },
            "model_loaded": False,
            "model_scored": False,
            "duration_seconds": round(time.perf_counter() - start_clock, 3),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(output_root / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except Exception as error:
        failure = {
            "status": "INCOMPLETE_DO_NOT_USE",
            "experiment_id": config["experiment_id"],
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(output_root / "attempt-failure.json", failure)
        _write_json(sealed_root / "attempt-failure.json", failure)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
