"""Integrity and metric helpers for a frozen MiniWoB prospective evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SEED_RE = re.compile(r"seed[-_](\d+)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def _add_seed_window(seeds: set[int], value: Any, *, inclusive_end: bool) -> None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(type(item) is int for item in value)
        and value[1] >= value[0]
    ):
        start, stop = value
        seeds.update(range(start, stop + int(inclusive_end)))


def _walk_metadata(value: Any, seeds: set[int], pixel_hashes: set[str]) -> None:
    if isinstance(value, dict):
        seed = value.get("seed")
        if type(seed) is int:
            seeds.add(seed)
        for key in ("screenshot_pixel_sha256", "pixel_sha256"):
            digest = value.get(key)
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                pixel_hashes.add(digest)
        _add_seed_window(seeds, value.get("seed_window_half_open"), inclusive_end=False)
        # Older smoke manifests used an inclusive-looking `seed_window`.
        _add_seed_window(seeds, value.get("seed_window"), inclusive_end=True)
        if type(value.get("seed_start")) is int and type(value.get("last_seed")) is int:
            _add_seed_window(seeds, [value["seed_start"], value["last_seed"]], inclusive_end=True)
        for child in value.values():
            _walk_metadata(child, seeds, pixel_hashes)
    elif isinstance(value, list):
        for child in value:
            _walk_metadata(child, seeds, pixel_hashes)


def previous_usage(roots: list[Path], explicit_seeds: list[int] | None = None) -> dict[str, Any]:
    """Read old MiniWoB receipts/screens and summarize all recorded seed use."""
    from tiny_omni_decision.miniwob_criterion_corpus import masked_screen
    from tiny_omni_decision.miniwob_criterion_corpus import pixel_sha256 as masked_pixel_sha256

    seeds = set(explicit_seeds or [])
    pixel_hashes: set[str] = set()
    masked_pixel_hashes: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for root in roots:
        root = root.resolve(strict=True)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        receipts.append({"root": str(root), "file_count": len(files)})
        for path in files:
            for match in _SEED_RE.finditer(path.name):
                seeds.add(int(match.group(1)))
            if path.suffix.lower() == ".png":
                with Image.open(path) as image:
                    pixel_hashes.add(pixel_sha256(image))
                    if image.size == (160, 210):
                        masked_pixel_hashes.add(masked_pixel_sha256(masked_screen(image)))
                continue
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            try:
                if path.suffix.lower() == ".jsonl":
                    values = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                else:
                    values = [json.loads(path.read_text(encoding="utf-8"))]
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"unreadable historical receipt: {path}") from exc
            for value in values:
                _walk_metadata(value, seeds, pixel_hashes)
            receipts[-1].setdefault("metadata_file_sha256", []).append(
                {"path": str(path), "sha256": sha256_file(path)}
            )
    ordered_seeds = sorted(seeds)
    ordered_pixels = sorted(pixel_hashes)
    return {
        "roots": receipts,
        "seed_count": len(ordered_seeds),
        "seed_values": ordered_seeds,
        "seed_values_sha256": hashlib.sha256(
            ",".join(map(str, ordered_seeds)).encode()
        ).hexdigest(),
        "screenshot_pixel_count": len(ordered_pixels),
        "screenshot_pixel_sha256": hashlib.sha256("\n".join(ordered_pixels).encode()).hexdigest(),
        "pixel_hashes": ordered_pixels,
        "masked_pixel_count": len(masked_pixel_hashes),
        "masked_pixel_sha256": hashlib.sha256(
            "\n".join(sorted(masked_pixel_hashes)).encode()
        ).hexdigest(),
        "masked_pixel_hashes": sorted(masked_pixel_hashes),
    }


def assert_seed_window_fresh(seed_window: list[int], previous_seeds: list[int]) -> None:
    if len(seed_window) != 2 or type(seed_window[0]) is not int or type(seed_window[1]) is not int:
        raise ValueError("seed window must contain two integers")
    start, stop = seed_window
    if start < 0 or stop <= start:
        raise ValueError("seed window must be a nonempty increasing half-open interval")
    overlap = [seed for seed in previous_seeds if start <= seed < stop]
    if overlap:
        raise ValueError(f"new seed window overlaps {len(overlap)} previously used seeds")


def validate_new_rows(
    rows: list[dict[str, Any]],
    *,
    corpus_root: Path,
    seed_window: list[int],
    expected_count: int,
    previous_seeds: list[int],
    previous_pixel_hashes: list[str],
) -> dict[str, Any]:
    from miniwob_screen_action_data import validate_screen_action_row

    assert_seed_window_fresh(seed_window, previous_seeds)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(rows)}")
    previous = set(previous_pixel_hashes)
    seen_seeds: set[int] = set()
    seen_groups: set[str] = set()
    seen_pixels: set[str] = set()
    for row in rows:
        validate_screen_action_row(row)
        seed = row["seed"]
        if not seed_window[0] <= seed < seed_window[1]:
            raise ValueError("new example seed is outside the preregistered window")
        if seed in previous_seeds or seed in seen_seeds:
            raise ValueError("new corpus contains a previously used or duplicate seed")
        if row["episode_group_id"] in seen_groups:
            raise ValueError("new corpus contains duplicate episode groups")
        path = (corpus_root / row["screenshot_path"]).resolve(strict=True)
        if not path.is_relative_to(corpus_root.resolve(strict=True)):
            raise ValueError("screenshot path escaped the prospective corpus")
        if sha256_file(path) != row["screenshot_sha256"]:
            raise ValueError("screenshot file hash differs from its row receipt")
        with Image.open(path) as image:
            digest = pixel_sha256(image)
            if list(np.asarray(image.convert("RGB")).shape) != row["screenshot_shape"]:
                raise ValueError("screenshot dimensions differ from its row receipt")
        if digest in previous:
            raise ValueError("new screenshot pixels overlap a previous MiniWoB artifact")
        if digest in seen_pixels:
            raise ValueError("new corpus contains duplicate screenshot pixels")
        seen_seeds.add(seed)
        seen_groups.add(row["episode_group_id"])
        seen_pixels.add(digest)
    return {
        "status": "PASS",
        "rows": len(rows),
        "unique_seeds": len(seen_seeds),
        "unique_episode_groups": len(seen_groups),
        "unique_screenshot_pixels": len(seen_pixels),
        "previous_seed_overlap": 0,
        "previous_screenshot_pixel_overlap": 0,
        "sample_order_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in rows).encode()
        ).hexdigest(),
        "screenshot_pixel_set_sha256": hashlib.sha256(
            "\n".join(sorted(seen_pixels)).encode()
        ).hexdigest(),
    }


def target_colors_by_sample_id(
    rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]
) -> dict[str, str]:
    """Join collector-only color labels onto model rows by the stable simulator seed."""
    accepted = [row for row in audit_rows if row.get("accepted")]
    color_by_seed: dict[int, str] = {}
    for row in accepted:
        seed = row.get("seed")
        color = row.get("target_color")
        if type(seed) is not int or not isinstance(color, str) or not color:
            raise ValueError(
                "accepted collector audit rows require an integer seed and target color"
            )
        if seed in color_by_seed:
            raise ValueError("collector audit contains duplicate accepted seeds")
        color_by_seed[seed] = color
    if len(rows) != len(accepted):
        raise ValueError("collector audit accepted-row count differs from model rows")
    colors: dict[str, str] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        seed = row.get("seed")
        if not isinstance(sample_id, str) or type(seed) is not int or seed not in color_by_seed:
            raise ValueError("model row cannot be joined to collector audit by seed")
        if sample_id in colors:
            raise ValueError("model rows contain duplicate sample IDs")
        colors[sample_id] = color_by_seed[seed]
    return colors


def metric_arrays(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    if (
        probabilities.ndim != 2
        or targets.ndim != 1
        or probabilities.shape[0] != targets.shape[0]
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(targets < 0)
        or np.any(targets >= probabilities.shape[1])
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("finite normalized probability rows and targets required")
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == targets
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (confidence <= upper if index == 14 else confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "count": int(targets.shape[0]),
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }


def _ece_bootstrap_samples(
    probabilities: np.ndarray, targets: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    sampled_probabilities = probabilities[samples]
    sampled_targets = targets[samples]
    confidence = sampled_probabilities.max(axis=2)
    correct = sampled_probabilities.argmax(axis=2) == sampled_targets
    ece = np.zeros(samples.shape[0], dtype=np.float64)
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        mask = (confidence >= lower) & (confidence <= upper if index == 14 else confidence < upper)
        counts = mask.sum(axis=1)
        accuracy = np.divide(
            (correct & mask).sum(axis=1), counts, out=np.zeros_like(ece), where=counts > 0
        )
        mean_confidence = np.divide(
            (confidence * mask).sum(axis=1), counts, out=np.zeros_like(ece), where=counts > 0
        )
        ece += (counts / samples.shape[1]) * np.abs(accuracy - mean_confidence)
    return ece


def episode_bootstrap(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 17,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    count = len(targets)
    correct = probabilities.argmax(axis=1) == targets
    row_nll = -np.log(np.clip(probabilities[np.arange(count), targets], 1e-12, 1))
    row_brier = np.square(probabilities - np.eye(probabilities.shape[1])[targets]).sum(axis=1)
    samples = rng.integers(0, count, size=(repetitions, count), endpoint=False)
    ece = _ece_bootstrap_samples(probabilities, targets, samples)
    return {
        "accuracy_95_percentile_ci": np.percentile(
            correct[samples].mean(axis=1), [2.5, 97.5]
        ).tolist(),
        "nll_95_percentile_ci": np.percentile(row_nll[samples].mean(axis=1), [2.5, 97.5]).tolist(),
        "brier_95_percentile_ci": np.percentile(
            row_brier[samples].mean(axis=1), [2.5, 97.5]
        ).tolist(),
        "ece_15_bins_95_percentile_ci": np.percentile(ece, [2.5, 97.5]).tolist(),
        "repetitions": repetitions,
        "seed": seed,
        "unit": "independent MiniWoB episode group; one row per group",
    }


def paired_bootstrap_difference(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 1701,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    count = len(targets)
    indices = rng.integers(0, count, size=(repetitions, count), endpoint=False)
    correct_a = probabilities_a.argmax(axis=1) == targets
    correct_b = probabilities_b.argmax(axis=1) == targets
    nll_a = -np.log(np.clip(probabilities_a[np.arange(count), targets], 1e-12, 1))
    nll_b = -np.log(np.clip(probabilities_b[np.arange(count), targets], 1e-12, 1))
    brier_a = np.square(probabilities_a - np.eye(probabilities_a.shape[1])[targets]).sum(axis=1)
    brier_b = np.square(probabilities_b - np.eye(probabilities_b.shape[1])[targets]).sum(axis=1)
    differences = {
        "accuracy_a_minus_b_95_percentile_ci": (
            correct_a[indices].astype(np.float64) - correct_b[indices].astype(np.float64)
        ).mean(axis=1),
        "nll_a_minus_b_95_percentile_ci": (nll_a[indices] - nll_b[indices]).mean(axis=1),
        "brier_a_minus_b_95_percentile_ci": (brier_a[indices] - brier_b[indices]).mean(axis=1),
    }
    ece_a = _ece_bootstrap_samples(probabilities_a, targets, indices)
    ece_b = _ece_bootstrap_samples(probabilities_b, targets, indices)
    differences["ece_15_bins_a_minus_b_95_percentile_ci"] = ece_a - ece_b
    return {key: np.percentile(values, [2.5, 97.5]).tolist() for key, values in differences.items()}
