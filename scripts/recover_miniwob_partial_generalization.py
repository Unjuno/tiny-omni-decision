"""Rebuild missing rows for a failed MiniWoB freeze without rewriting screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tiny_omni_decision.miniwob_generalization import (
    assert_seed_window_fresh,
    previous_usage,
    sha256_file,
    validate_new_rows,
)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path)
    parser.add_argument("--browser-binary", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    partial = (args.partial_dir or Path(config["prospective_corpus"]["output"])).resolve(
        strict=True
    )
    manifest_path = partial / "manifest.json"
    rows_path = partial / "evaluation.jsonl"
    audit_path = partial / "collector-audit.jsonl"
    recovery_path = partial / "recovery-receipt.json"
    if any(path.exists() for path in (manifest_path, rows_path, audit_path, recovery_path)):
        raise FileExistsError("refusing to overwrite a partially or fully finalized corpus")

    preflight_path = partial / "pre-generation-preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    if preflight.get("config_sha256") != config_sha:
        raise ValueError("partial corpus was generated under a different frozen config")
    if preflight.get("status") != "PASS_BEFORE_GENERATION":
        raise ValueError("partial corpus lacks a passing pre-generation gate")

    historical_roots = [Path(value) for value in config["historical_artifacts"]]
    prior = previous_usage(
        historical_roots, config.get("historical_seeds_without_saved_corpus", [])
    )
    usage_receipt = preflight["historical_usage"]
    if (
        prior["seed_values_sha256"] != usage_receipt["seed_values_sha256"]
        or prior["screenshot_pixel_sha256"] != usage_receipt["screenshot_pixel_sha256"]
    ):
        raise ValueError("historical use inventory changed after the partial collection")

    seed_window = config["prospective_corpus"]["seed_window_half_open"]
    assert_seed_window_fresh(seed_window, prior["seed_values"])
    expected_count = config["prospective_corpus"]["accepted_episode_count"]
    split_dir = partial / "evaluation"
    existing_files = sorted(split_dir.glob("*.png"))
    if len(existing_files) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} preserved screenshots, found {len(existing_files)}"
        )
    for path in existing_files:
        name = path.stem
        if not name.startswith("evaluation-seed-"):
            raise ValueError(f"unexpected preserved screenshot name: {path.name}")
        seed = int(name.removeprefix("evaluation-seed-"))
        if not seed_window[0] <= seed < seed_window[1]:
            raise ValueError(f"preserved screenshot seed is outside the frozen window: {seed}")

    source_root = Path(config["source"]["root"]).resolve(strict=True)
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != config["source"]["commit"]:
        raise ValueError("MiniWoB++ checkout is not at the pinned commit")
    license_path = source_root / "LICENSE"
    if sha256_file(license_path) != config["source"]["license_sha256"]:
        raise ValueError("MiniWoB++ license hash differs from the frozen receipt")
    if sha256_file(args.driver.resolve(strict=True)) != preflight["driver_sha256"]:
        raise ValueError("recovery driver differs from the preflight-verified driver")

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    import freeze_miniwob_screen_action_corpus as legacy_collector
    from miniwob_screen_action_data import write_canonical_jsonl

    browser_binary = args.browser_binary.resolve(strict=True)
    driver = args.driver.resolve(strict=True)
    collector_args = SimpleNamespace(edge_compat=True, browser_binary=browser_binary, driver=driver)
    legacy_collector._configure_edge(collector_args)
    os.environ["MINIWOB_CHROME_BINARY"] = str(browser_binary)
    os.environ["MINIWOB_CHROMEDRIVER"] = str(driver)
    sys.path.insert(0, str(source_root))
    import gymnasium
    import miniwob

    gymnasium.register_envs(miniwob)
    env = gymnasium.make("miniwob/click-color-v1", render_mode=None, wait_ms=200)
    try:
        rows, audit, used_seeds = legacy_collector._collect_split(
            env=env,
            output_dir=partial,
            split="evaluation",
            seed_range=tuple(seed_window),
            count=expected_count,
            seen_screen_hashes=set(prior["pixel_hashes"]),
            reuse_existing_screenshots=True,
        )
    finally:
        env.close()

    expected_names = {path.name for path in existing_files}
    replayed_names = {Path(row["screenshot_path"]).name for row in rows}
    if replayed_names != expected_names or len(used_seeds) != expected_count:
        raise ValueError("deterministic replay did not reproduce the exact preserved cohort")
    integrity = validate_new_rows(
        rows,
        corpus_root=partial,
        seed_window=seed_window,
        expected_count=expected_count,
        previous_seeds=prior["seed_values"],
        previous_pixel_hashes=prior["pixel_hashes"],
    )

    integrity.update(
        {
            "status": "FROZEN_PROSPECTIVE_EVALUATION_NO_MODEL_SCORING",
            "experiment_id": config["experiment_id"],
            "config_sha256": config_sha,
            "source_repository": config["source"]["repository"],
            "source_commit": source_commit,
            "source_license": config["source"]["license"],
            "source_license_sha256": sha256_file(license_path),
            "task": config["task"],
            "seed_window_half_open": seed_window,
            "seed_order_sha256": hashlib.sha256(
                ",".join(map(str, used_seeds)).encode()
            ).hexdigest(),
            "attempted_seed_count": len(audit),
            "accepted_seed_count": len(used_seeds),
            "examples_sha256": write_canonical_jsonl(rows_path, rows),
            "collector_audit_sha256": write_canonical_jsonl(audit_path, audit),
            "historical_usage": usage_receipt,
            "model_loaded": False,
            "model_scored": False,
            "sealed_audit": False,
            "recovery": {
                "reason": (
                    "Original post-collection finalization failed on a missing module import."
                ),
                "replayed_seed_window": True,
                "reused_preserved_screenshots_without_overwriting": True,
                "preserved_screenshot_count": len(existing_files),
                "screenshot_pixel_identity_verified": True,
                "prior_reconstruction_failed_attempt": str(partial),
            },
        }
    )
    _write_new_json(manifest_path, integrity)
    _write_new_json(
        recovery_path,
        {
            "status": "RECOVERED_WITH_EXACT_PRESERVED_SCREEN_PIXELS",
            "config_sha256": config_sha,
            "preflight_sha256": sha256_file(preflight_path),
            "manifest_sha256": sha256_file(manifest_path),
            "evaluation_jsonl_sha256": sha256_file(rows_path),
            "collector_audit_sha256": sha256_file(audit_path),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "model_loaded": False,
            "model_scored": False,
        },
    )
    print(
        json.dumps(
            {
                "status": integrity["status"],
                "manifest_sha256": sha256_file(manifest_path),
                "evaluation_jsonl_sha256": integrity["examples_sha256"],
                "rows": integrity["rows"],
                "unique_episode_groups": integrity["unique_episode_groups"],
                "previous_seed_overlap": integrity["previous_seed_overlap"],
                "previous_screenshot_pixel_overlap": integrity["previous_screenshot_pixel_overlap"],
                "model_scored": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
