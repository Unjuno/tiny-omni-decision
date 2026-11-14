"""Freeze one new screen-only MiniWoB cohort; this script never loads a model."""

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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--browser-binary", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_cfg = config["prospective_corpus"]
    seed_window = corpus_cfg["seed_window_half_open"]
    output = (args.output_dir or Path(corpus_cfg["output"])).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prospective corpus: {output}")

    roots = [Path(value) for value in config["historical_artifacts"]]
    prior = previous_usage(roots, config.get("historical_seeds_without_saved_corpus", []))
    assert_seed_window_fresh(seed_window, prior["seed_values"])

    source_root = Path(config["source"]["root"]).resolve(strict=True)
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != config["source"]["commit"]:
        raise ValueError("MiniWoB++ checkout is not at the preregistered commit")
    license_path = source_root / "LICENSE"
    if sha256_file(license_path) != config["source"]["license_sha256"]:
        raise ValueError("MiniWoB++ license receipt differs from the pinned hash")

    output.mkdir(parents=True)
    preflight = {
        "status": "PASS_BEFORE_GENERATION",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "seed_window_half_open": seed_window,
        "accepted_episode_count": corpus_cfg["accepted_episode_count"],
        "historical_usage": {
            "roots": prior["roots"],
            "seed_count": prior["seed_count"],
            "seed_values_sha256": prior["seed_values_sha256"],
            "screenshot_pixel_count": prior["screenshot_pixel_count"],
            "screenshot_pixel_sha256": prior["screenshot_pixel_sha256"],
            "seed_window_overlap": 0,
        },
        "source_commit": source_commit,
        "source_license_sha256": sha256_file(license_path),
        "browser_binary": str(args.browser_binary.resolve(strict=True)),
        "driver": str(args.driver.resolve(strict=True)),
        "driver_sha256": sha256_file(args.driver.resolve(strict=True)),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(output / "pre-generation-preflight.json", preflight)

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    import freeze_miniwob_screen_action_corpus as legacy_collector
    from miniwob_screen_action_data import write_canonical_jsonl

    collector_args = SimpleNamespace(
        edge_compat=True,
        browser_binary=args.browser_binary,
        driver=args.driver,
    )
    legacy_collector._configure_edge(collector_args)
    os.environ["MINIWOB_CHROME_BINARY"] = str(args.browser_binary.resolve(strict=True))
    os.environ["MINIWOB_CHROMEDRIVER"] = str(args.driver.resolve(strict=True))
    sys.path.insert(0, str(source_root))
    import gymnasium
    import miniwob

    gymnasium.register_envs(miniwob)
    env = gymnasium.make("miniwob/click-color-v1", render_mode=None, wait_ms=200)
    try:
        rows, audit, used_seeds = legacy_collector._collect_split(
            env=env,
            output_dir=output,
            split="evaluation",
            seed_range=tuple(seed_window),
            count=corpus_cfg["accepted_episode_count"],
            seen_screen_hashes=set(prior["pixel_hashes"]),
        )
    finally:
        env.close()

    manifest = validate_new_rows(
        rows,
        corpus_root=output,
        seed_window=seed_window,
        expected_count=corpus_cfg["accepted_episode_count"],
        previous_seeds=prior["seed_values"],
        previous_pixel_hashes=prior["pixel_hashes"],
    )
    manifest.update(
        {
            "status": "FROZEN_PROSPECTIVE_EVALUATION_NO_MODEL_SCORING",
            "experiment_id": config["experiment_id"],
            "config_sha256": sha256_file(config_path),
            "source_repository": config["source"]["repository"],
            "source_commit": source_commit,
            "source_license": config["source"]["license"],
            "source_license_sha256": sha256_file(license_path),
            "task": config["task"],
            "seed_window_half_open": seed_window,
            "attempted_seed_count": len(audit),
            "accepted_seed_count": len(used_seeds),
            "seed_order_sha256": hashlib.sha256(
                ",".join(map(str, used_seeds)).encode()
            ).hexdigest(),
            "examples_sha256": write_canonical_jsonl(output / "evaluation.jsonl", rows),
            "collector_audit_sha256": write_canonical_jsonl(
                output / "collector-audit.jsonl", audit
            ),
            "historical_usage": preflight["historical_usage"],
            "model_loaded": False,
            "model_scored": False,
            "sealed_audit": False,
        }
    )
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "pre-generation-preflight.json", preflight)
    print(json.dumps({"status": manifest["status"], "output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
