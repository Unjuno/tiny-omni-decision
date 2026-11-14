"""One-shot CPU inference on the frozen prospective MiniWoB cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from miniwob_screen_action_data import model_option_positions

from tiny_omni_decision.miniwob_dino_readout import deranged_indices
from tiny_omni_decision.miniwob_dino_runtime import MiniWoBDinoDecisionRuntime
from tiny_omni_decision.miniwob_generalization import (
    episode_bootstrap,
    metric_arrays,
    paired_bootstrap_difference,
    previous_usage,
    sha256_file,
    target_colors_by_sample_id,
    validate_new_rows,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_options(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": option["option_id"],
            # The runtime returns this display string but does not pass it to DINO or the scorer.
            "text": f"Action candidate {index}",
            "x_norm": option["x_norm"],
            "y_norm": option["y_norm"],
        }
        for index, option in enumerate(row["options"])
    ]


def _probabilities_by_id(
    runtime: MiniWoBDinoDecisionRuntime,
    *,
    image: Image.Image,
    row: dict[str, Any],
    options: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    request_options = options if options is not None else _row_options(row)
    response = runtime.predict(request_id=row["sample_id"], image=image, options=request_options)
    output_by_id = {item["id"]: float(item["probability"]) for item in response["options"]}
    wanted_ids = [option["option_id"] for option in row["options"]]
    values = np.asarray([output_by_id[option_id] for option_id in wanted_ids], dtype=np.float64)
    if not np.isfinite(values).all() or not np.isclose(values.sum(), 1.0, atol=1e-6):
        raise ValueError(f"invalid probabilities for {row['sample_id']}")
    return values


def _position_prior(train_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> np.ndarray:
    counts: Counter[tuple[float, float]] = Counter()
    for row in train_rows:
        counts[model_option_positions(row)[row["target_index"]]] += 1
    rates = {position: counts[position] / len(train_rows) for position in counts}
    output: list[np.ndarray] = []
    for row in rows:
        logits = np.asarray(
            [
                np.log(max(rates.get(position, 0.0), 1e-6))
                for position in model_option_positions(row)
            ],
            dtype=np.float64,
        )
        logits -= logits.max()
        exp_logits = np.exp(logits)
        output.append(exp_logits / exp_logits.sum())
    return np.stack(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve(strict=True)
    config_sha = sha256_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_root = (args.corpus_dir or Path(config["prospective_corpus"]["output"])).resolve(
        strict=True
    )
    output = (args.output_dir or Path(config["output"])).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")

    corpus_manifest_path = corpus_root / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "FROZEN_PROSPECTIVE_EVALUATION_NO_MODEL_SCORING":
        raise ValueError("prospective corpus is not frozen for its one-time evaluation")
    preflight = json.loads(
        (corpus_root / "pre-generation-preflight.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "PASS_BEFORE_GENERATION":
        raise ValueError("pre-generation independence gate did not pass")
    if corpus_manifest.get("config_sha256") != config_sha:
        raise ValueError("corpus config hash does not match the frozen experiment config")

    roots = [Path(value) for value in config["historical_artifacts"]]
    prior = previous_usage(roots, config.get("historical_seeds_without_saved_corpus", []))
    if (
        prior["seed_values_sha256"] != corpus_manifest["historical_usage"]["seed_values_sha256"]
        or prior["screenshot_pixel_sha256"]
        != corpus_manifest["historical_usage"]["screenshot_pixel_sha256"]
    ):
        raise ValueError("historical MiniWoB artifact inventory changed after generation")

    rows = _read_jsonl(corpus_root / "evaluation.jsonl")
    audit_rows = _read_jsonl(corpus_root / "collector-audit.jsonl")
    if [row["seed"] for row in rows] != [row["seed"] for row in audit_rows if row["accepted"]]:
        raise ValueError("collector audit and model-visible row order differ")
    integrity = validate_new_rows(
        rows,
        corpus_root=corpus_root,
        seed_window=config["prospective_corpus"]["seed_window_half_open"],
        expected_count=config["prospective_corpus"]["accepted_episode_count"],
        previous_seeds=prior["seed_values"],
        previous_pixel_hashes=prior["pixel_hashes"],
    )
    if sha256_file(corpus_root / "evaluation.jsonl") != corpus_manifest["examples_sha256"]:
        raise ValueError("prospective row JSONL hash differs from frozen manifest")

    legacy_root = Path(config["legacy_training_corpus"]["root"]).resolve(strict=True)
    if (
        sha256_file(legacy_root / "manifest.json")
        != config["legacy_training_corpus"]["manifest_sha256"]
    ):
        raise ValueError("legacy train corpus manifest changed")
    train_path = legacy_root / "train.jsonl"
    if sha256_file(train_path) != config["legacy_training_corpus"]["train_jsonl_sha256"]:
        raise ValueError("legacy train rows changed")
    # Only the old TRAIN rows are read to compute the preregistered position prior.
    train_rows = _read_jsonl(train_path)

    encoder_cfg, readout_cfg = config["encoder"], config["readout"]
    encoder_dir = Path(encoder_cfg["directory"]).resolve(strict=True)
    readout_path = Path(readout_cfg["path"]).resolve(strict=True)
    if sha256_file(encoder_dir / "model.safetensors") != encoder_cfg["weight_sha256"]:
        raise ValueError("pinned DINOv2 weights changed")
    if (
        sha256_file(encoder_dir / "preprocessor_config.json")
        != encoder_cfg["processor_config_sha256"]
    ):
        raise ValueError("pinned DINOv2 processor config changed")
    if sha256_file(readout_path) != readout_cfg["sha256"]:
        raise ValueError("selected readout checkpoint changed")
    if config["device"] != "cpu":
        raise ValueError("this inference-only protocol is fixed to CPU")

    start_utc = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    torch.set_num_threads(2)
    runtime = MiniWoBDinoDecisionRuntime.from_local_artifacts(
        encoder_dir=encoder_dir,
        readout_path=readout_path,
        encoder_sha256=encoder_cfg["weight_sha256"],
        readout_sha256=readout_cfg["sha256"],
        device="cpu",
    )
    if any(parameter.requires_grad for parameter in runtime.encoder.parameters()):
        raise ValueError("encoder unexpectedly has trainable parameters")
    targets = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    images: list[Image.Image] = []
    for row in rows:
        image_path = (corpus_root / row["screenshot_path"]).resolve(strict=True)
        images.append(Image.open(image_path).convert("RGB"))

    scenario_probabilities: dict[str, np.ndarray] = {}
    scenario_durations: dict[str, float] = {}
    scenarios = {
        "matched_screen": images,
        "sattolo_shuffled_screen": [images[index] for index in deranged_indices(len(images), 1701)],
        "black_screen": [Image.new("RGB", image.size, (0, 0, 0)) for image in images],
    }
    for scenario, scenario_images in scenarios.items():
        started = time.perf_counter()
        predictions: list[np.ndarray] = []
        for index, (row, image) in enumerate(zip(rows, scenario_images, strict=True), start=1):
            predictions.append(_probabilities_by_id(runtime, image=image, row=row))
            if index % 128 == 0:
                print(f"{scenario}: {index}/{len(rows)}", flush=True)
        scenario_probabilities[scenario] = np.stack(predictions)
        scenario_durations[scenario] = time.perf_counter() - started

    prior_probabilities = _position_prior(train_rows, rows)
    scenario_probabilities["train_position_prior"] = prior_probabilities
    option_order = config["controls"]["option_order_permutation"]
    permuted_predictions: list[np.ndarray] = []
    permutation_deltas: list[float] = []
    started = time.perf_counter()
    for index, (row, image) in enumerate(zip(rows, images, strict=True), start=1):
        options = _row_options(row)
        permuted_options = [options[position] for position in option_order]
        raw = _probabilities_by_id(runtime, image=image, row=row, options=permuted_options)
        # _probabilities_by_id restores by option ID to the original row order.
        permuted_predictions.append(raw)
        permutation_deltas.append(
            float(np.max(np.abs(raw - scenario_probabilities["matched_screen"][index - 1])))
        )
        if index % 128 == 0:
            print(f"candidate_order: {index}/{len(rows)}", flush=True)
    scenario_probabilities["candidate_order_remapped"] = np.stack(permuted_predictions)
    scenario_durations["candidate_order_remapped"] = time.perf_counter() - started

    color_by_id = target_colors_by_sample_id(rows, audit_rows)
    results: dict[str, Any] = {
        "status": "COMPLETE_FROZEN_CHECKPOINT_PROSPECTIVE_EVALUATION",
        "scope": (
            "synthetic MiniWoB click-color screen-only task; development/generalization evidence, "
            "not product audit"
        ),
        "count": len(rows),
        "unique_episode_groups": len({row["episode_group_id"] for row in rows}),
        "seed_order_sha256": corpus_manifest["seed_order_sha256"],
        "sample_order_sha256": integrity["sample_order_sha256"],
        "metrics_by_scenario": {
            name: metric_arrays(probabilities, targets)
            for name, probabilities in scenario_probabilities.items()
        },
        "matched_screen_episode_bootstrap": episode_bootstrap(
            scenario_probabilities["matched_screen"],
            targets,
            repetitions=config["metrics"]["episode_bootstrap_repetitions"],
            seed=config["metrics"]["episode_bootstrap_seed"],
        ),
        "paired_episode_bootstrap_matched_minus_control": {
            name: paired_bootstrap_difference(
                scenario_probabilities["matched_screen"],
                probabilities,
                targets,
                repetitions=config["metrics"]["paired_bootstrap_repetitions"],
                seed=config["metrics"]["paired_bootstrap_seed"],
            )
            for name, probabilities in scenario_probabilities.items()
            if name != "matched_screen"
        },
        "matched_screen_accuracy_by_target_color": {
            color: metric_arrays(
                scenario_probabilities["matched_screen"][
                    np.asarray([color_by_id[row["sample_id"]] == color for row in rows])
                ],
                targets[np.asarray([color_by_id[row["sample_id"]] == color for row in rows])],
            )
            for color in sorted(set(color_by_id.values()))
        },
        "candidate_order_equivariance": {
            "permutation": option_order,
            "max_option_id_aligned_probability_abs_difference": max(permutation_deltas),
            "mean_option_id_aligned_probability_abs_difference": float(np.mean(permutation_deltas)),
            "tolerance": config["controls"]["option_order_max_probability_abs_tolerance"],
            "passed": max(permutation_deltas)
            <= config["controls"]["option_order_max_probability_abs_tolerance"],
        },
        "control_seeds": {"sattolo_screen_permutation": 1701},
        "no_training_or_calibration": True,
        "checkpoint_selection": False,
        "sealed_audit_loaded": False,
        "model": {
            "encoder_revision": encoder_cfg["revision"],
            "encoder_sha256": encoder_cfg["weight_sha256"],
            "encoder_parameters": sum(
                parameter.numel() for parameter in runtime.encoder.parameters()
            ),
            "encoder_trainable_parameters": sum(
                parameter.numel()
                for parameter in runtime.encoder.parameters()
                if parameter.requires_grad
            ),
            "readout_sha256": readout_cfg["sha256"],
            "readout_file_bytes": readout_path.stat().st_size,
            "readout_parameters": sum(
                parameter.numel() for parameter in runtime.scorer.parameters()
            ),
            "resident_parameters": sum(
                parameter.numel() for parameter in runtime.encoder.parameters()
            )
            + sum(parameter.numel() for parameter in runtime.scorer.parameters()),
            "temperature": readout_cfg["temperature"],
        },
        "runtime": {
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "wall_seconds_by_scenario": scenario_durations,
            "wall_seconds_total": time.perf_counter() - wall_start,
            "cpu_seconds_total": time.process_time() - cpu_start,
            "cloud_cost_usd": 0,
        },
    }
    if not results["candidate_order_equivariance"]["passed"]:
        raise ValueError("fixed scorer failed its predeclared candidate-order control")

    output.mkdir(parents=True)
    prediction_records: list[dict[str, Any]] = []
    scenario_names = list(scenario_probabilities)
    for index, row in enumerate(rows):
        by_scenario = {
            name: scenario_probabilities[name][index].tolist() for name in scenario_names
        }
        main = scenario_probabilities["matched_screen"][index]
        prediction_records.append(
            {
                "sample_id": row["sample_id"],
                "episode_group_id": row["episode_group_id"],
                "seed": row["seed"],
                "target_index": int(targets[index]),
                "target_color": color_by_id[row["sample_id"]],
                "prediction_index": int(main.argmax()),
                "probabilities_by_scenario": by_scenario,
            }
        )
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_bytes(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_records).encode(
            "utf-8"
        )
    )
    results["prediction_sha256"] = sha256_file(prediction_path)
    results["prediction_records_sha256"] = hashlib.sha256(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_records).encode(
            "utf-8"
        )
    ).hexdigest()
    results["started_at_utc"] = start_utc
    results["finished_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(output / "results.json", results)
    _write_json(
        output / "run-metadata.json",
        {
            "experiment_id": config["experiment_id"],
            "config_sha256": config_sha,
            "source_commit": subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "evaluation_jsonl_sha256": sha256_file(corpus_root / "evaluation.jsonl"),
            "pre_generation_preflight_sha256": sha256_file(
                corpus_root / "pre-generation-preflight.json"
            ),
            "result_sha256": sha256_file(output / "results.json"),
            "prediction_sha256": results["prediction_sha256"],
            "results": results,
        },
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "output": str(output),
                "metrics": results["metrics_by_scenario"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
