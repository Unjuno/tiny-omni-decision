"""One-shot held-category transfer probe from frozen AVATAR TRAIN features only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SEED = 17
HELD_CATEGORY_COUNT = 16
VALIDATION_FRACTION = 0.20
BOOTSTRAP_REPLICATES = 5_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{SEED}:{value}".encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def build_transfer_partition(
    queries: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    *,
    held_category_count: int = HELD_CATEGORY_COUNT,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    """Create disjoint source-group pools and a deterministic category holdout."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    categories = sorted({str(row["category"]) for row in queries})
    video_categories = {str(row["category"]) for row in videos}
    if set(categories) != video_categories:
        raise ValueError("TRAIN query and video category inventories differ")
    if not 4 <= held_category_count < len(categories):
        raise ValueError("held_category_count must leave at least four seen categories")
    held_categories = set(
        sorted(categories, key=lambda category: _rank("avatar-category-transfer-v1", category))[
            :held_category_count
        ]
    )
    group_categories: dict[str, set[str]] = defaultdict(set)
    for row in queries:
        group_categories[str(row["source_group"])].add(str(row["category"]))
    for row in videos:
        group_categories[str(row["source_group"])].add(str(row["category"]))
    held_groups = {group for group, values in group_categories.items() if values & held_categories}
    seen_groups = set(group_categories) - held_groups
    if len(seen_groups) < 2 or len(held_groups) < 2:
        raise ValueError("source-group quarantine left too few groups")
    validation_count = max(1, round(len(seen_groups) * validation_fraction))
    validation_groups = set(
        sorted(seen_groups, key=lambda group: _rank("avatar-inner-groups-v1", group))[
            :validation_count
        ]
    )
    fit_groups = seen_groups - validation_groups
    if fit_groups & validation_groups or (fit_groups | validation_groups) != seen_groups:
        raise AssertionError("seen source groups were not partitioned exactly")
    if (fit_groups | validation_groups) & held_groups:
        raise AssertionError("held and seen source groups overlap")
    return {
        "held_categories": sorted(held_categories),
        "seen_categories": sorted(set(categories) - held_categories),
        "fit_groups": fit_groups,
        "validation_groups": validation_groups,
        "held_test_groups": held_groups,
        "group_categories": group_categories,
    }


def build_choices(
    queries: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    *,
    split: str,
    allowed_groups: set[str],
    allowed_categories: set[str],
) -> list[dict[str, Any]]:
    """Build four-choice rows solely from a split's own groups and categories."""
    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in videos:
        category, group = str(row["category"]), str(row["source_group"])
        if category in allowed_categories and group in allowed_groups:
            pools[category][group].append(row)
    for group_map in pools.values():
        for group in group_map:
            group_map[group].sort(key=lambda row: str(row["asset_key"]))

    output = []
    seen_queries: set[str] = set()
    for query in sorted(queries, key=lambda row: str(row["query_id"])):
        query_id = str(query["query_id"])
        category = str(query["category"])
        query_group = str(query["source_group"])
        if category not in allowed_categories or query_group not in allowed_groups:
            continue
        if query_id in seen_queries:
            raise ValueError(f"duplicate query identity: {query_id}")
        seen_queries.add(query_id)
        positive_pool = [
            candidate
            for group, candidates in pools.get(category, {}).items()
            if group != query_group
            for candidate in candidates
            if str(candidate["asset_key"]) != str(query["asset_key"])
            and abs(float(candidate["timestamp_seconds"]) - float(query["timestamp_seconds"]))
            > 1e-9
        ]
        positive_pool.sort(
            key=lambda row: _rank(
                f"avatar-transfer:{split}:positive:{query_id}", str(row["asset_key"])
            )
        )
        if not positive_pool:
            continue
        positive = positive_pool[0]
        used_groups = {query_group, str(positive["source_group"])}
        negatives = []
        negative_categories = sorted(
            (value for value in pools if value != category),
            key=lambda value: _rank(f"avatar-transfer:{split}:negative-category:{query_id}", value),
        )
        for negative_category in negative_categories:
            candidate_pool = [
                candidate
                for group, candidates in pools[negative_category].items()
                if group not in used_groups
                for candidate in candidates
            ]
            if not candidate_pool:
                continue
            candidate_pool.sort(
                key=lambda row: _rank(
                    f"avatar-transfer:{split}:negative:{query_id}:{negative_category}",
                    str(row["asset_key"]),
                )
            )
            candidate = candidate_pool[0]
            negatives.append(candidate)
            used_groups.add(str(candidate["source_group"]))
            if len(negatives) == 3:
                break
        if len(negatives) != 3:
            continue
        candidates = [positive, *negatives]
        candidates.sort(
            key=lambda row: _rank(
                f"avatar-transfer:{split}:option-order:{query_id}", str(row["asset_key"])
            )
        )
        target = next(index for index, candidate in enumerate(candidates) if candidate is positive)
        output.append(
            {
                "query_id": query_id,
                "audio_asset_key": str(query["asset_key"]),
                "audio_source_group": query_group,
                "category": category,
                "audio_timestamp_seconds": float(query["timestamp_seconds"]),
                "video_asset_keys": [str(row["asset_key"]) for row in candidates],
                "video_source_groups": [str(row["source_group"]) for row in candidates],
                "option_categories": [str(row["category"]) for row in candidates],
                "video_timestamps_seconds": [float(row["timestamp_seconds"]) for row in candidates],
                "target_index": target,
                "split": split,
            }
        )
    return output


def validate_choice_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    group_sets = {}
    asset_sets = {}
    id_sets = {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"choice split is empty: {name}")
        groups = set()
        assets = set()
        ids = set()
        for row in rows:
            candidates = row["video_source_groups"]
            categories = row["option_categories"]
            if len(candidates) != 4 or len(set(candidates)) != 4:
                raise ValueError(f"candidate source groups are not unique: {row['query_id']}")
            if row["audio_source_group"] in candidates:
                raise ValueError(f"query source is reused as a candidate: {row['query_id']}")
            if len(set(categories)) != 4:
                raise ValueError(f"candidate categories are not unique: {row['query_id']}")
            if categories[row["target_index"]] != row["category"]:
                raise ValueError(f"positive category mismatch: {row['query_id']}")
            if row["video_asset_keys"][row["target_index"]] == row["audio_asset_key"]:
                raise ValueError(f"positive reuses the query asset: {row['query_id']}")
            row_groups = {row["audio_source_group"], *candidates}
            row_assets = {row["audio_asset_key"], *row["video_asset_keys"]}
            groups.update(row_groups)
            assets.update(row_assets)
            ids.add(row["query_id"])
        group_sets[name], asset_sets[name], id_sets[name] = groups, assets, ids
    names = list(splits)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            if group_sets[first] & group_sets[second]:
                raise ValueError(f"source-group overlap between {first} and {second}")
            if asset_sets[first] & asset_sets[second]:
                raise ValueError(f"asset overlap between {first} and {second}")
            if id_sets[first] & id_sets[second]:
                raise ValueError(f"query overlap between {first} and {second}")
    return {
        "source_group_counts": {name: len(value) for name, value in group_sets.items()},
        "asset_counts": {name: len(value) for name, value in asset_sets.items()},
        "query_counts": {name: len(rows) for name, rows in splits.items()},
        "cross_split_source_group_overlap": 0,
        "cross_split_asset_overlap": 0,
        "cross_split_query_overlap": 0,
    }


def _cluster_bootstrap_accuracy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    if len(probabilities) != len(targets) or len(groups) != len(targets):
        raise ValueError("bootstrap inputs are not aligned")
    by_group = {group: np.flatnonzero(np.asarray(groups) == group) for group in sorted(set(groups))}
    if len(by_group) < 2:
        raise ValueError("cluster bootstrap requires at least two groups")
    correct = probabilities.argmax(axis=1) == targets
    rng = np.random.default_rng(SEED)
    names = list(by_group)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(names), size=len(names))
        indices = np.concatenate([by_group[names[int(index)]] for index in sampled])
        values[replicate] = correct[indices].mean()
    return {
        "seed": SEED,
        "replicates": replicates,
        "resampling_unit": "audio query source group, sampled with replacement",
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _query_and_video_pools(
    rows: list[dict[str, Any]], sidecars: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != len(sidecars):
        raise ValueError("TRAIN model rows and sidecars are not aligned")
    sidecar_by_id = {str(row["example_id"]): row for row in sidecars}
    if len(sidecar_by_id) != len(sidecars):
        raise ValueError("duplicate TRAIN sidecar example ID")
    queries: dict[str, dict[str, Any]] = {}
    videos: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("non-TRAIN model input was supplied")
        example_id = str(row["example_id"])
        sidecar = sidecar_by_id.pop(example_id, None)
        if sidecar is None:
            raise ValueError(f"missing TRAIN sidecar: {example_id}")
        audio_key = str(row["audio_asset_key"])
        if audio_key[:11] != str(sidecar["audio_source_group"]):
            raise ValueError(f"audio asset/source group mismatch: {audio_key}")
        query = {
            "query_id": audio_key,
            "asset_key": audio_key,
            "source_group": str(sidecar["audio_source_group"]),
            "category": str(sidecar["target_category"]),
            "timestamp_seconds": float(row["audio_window_center_seconds"]),
            "duration_seconds": float(row["audio_window_duration_seconds"]),
            "left_padding_seconds": float(row["audio_window_left_padding_seconds"]),
            "right_padding_seconds": float(row["audio_window_right_padding_seconds"]),
        }
        previous = queries.setdefault(audio_key, query)
        if previous != query:
            raise ValueError(
                f"one audio asset has conflicting TRAIN query labels/metadata: {audio_key}"
            )
        for index, option in enumerate(row["options"]):
            key = str(option["video_asset_key"])
            candidate = {
                "asset_key": key,
                "source_group": str(sidecar["option_source_groups"][index]),
                "category": str(sidecar["option_categories"][index]),
                "timestamp_seconds": float(option["video_timestamp_seconds"]),
            }
            if key[:11] != candidate["source_group"]:
                raise ValueError(f"video asset/source group mismatch: {key}")
            old = videos.setdefault(key, candidate)
            if old != candidate:
                raise ValueError(f"one video asset has conflicting TRAIN metadata: {key}")
    if sidecar_by_id:
        raise ValueError(f"unused TRAIN sidecars: {len(sidecar_by_id)}")
    return list(queries.values()), list(videos.values())


def _load_train_features(
    path: Path, required_keys: set[str], *, kind: str, dimension: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        all_keys = [str(value) for value in archive["asset_keys"].tolist()]
        if len(set(all_keys)) != len(all_keys):
            raise ValueError(f"duplicate IDs in {kind} feature cache")
        index = {key: position for position, key in enumerate(all_keys)}
        missing = required_keys - index.keys()
        if missing:
            raise ValueError(f"{kind} cache is missing {len(missing)} TRAIN feature IDs")
        selected_keys = sorted(required_keys)
        positions = [index[key] for key in selected_keys]
        features = np.asarray(archive["features"][positions], dtype=np.float32)
        if features.shape != (len(selected_keys), dimension) or not np.isfinite(features).all():
            raise ValueError(f"invalid selected TRAIN {kind} features")
        extra_names = {
            "audio": ("centers_seconds", "declared_padding_samples"),
            "video": ("timestamps",),
        }[kind]
        extra = {name: np.asarray(archive[name][positions]) for name in extra_names}
    if kind == "audio":
        if not np.isfinite(extra["centers_seconds"]).all():
            raise ValueError("non-finite selected TRAIN audio timestamps")
        if extra["declared_padding_samples"].shape != (len(selected_keys), 2):
            raise ValueError("invalid selected TRAIN audio padding metadata")
    elif not np.isfinite(extra["timestamps"]).all():
        raise ValueError("non-finite selected TRAIN video timestamps")
    return (
        {key: features[index] for index, key in enumerate(selected_keys)},
        {
            "train_asset_count": len(selected_keys),
            "feature_shape": list(features.shape),
            "all_cache_asset_count_metadata_only": len(all_keys),
            "selected_features_finite": True,
            "selected_metadata_sha256": hashlib.sha256(
                json.dumps(
                    {name: value.tolist() for name, value in extra.items()}, sort_keys=True
                ).encode()
            ).hexdigest(),
        },
    )


def _arrays(
    rows: list[dict[str, Any]],
    audio_features: dict[str, np.ndarray],
    video_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    audio, video, targets, groups, ids = [], [], [], [], []
    for row in rows:
        audio.append(audio_features[row["audio_asset_key"]])
        video.append(np.stack([video_features[key] for key in row["video_asset_keys"]]))
        targets.append(int(row["target_index"]))
        groups.append(str(row["audio_source_group"]))
        ids.append(str(row["query_id"]))
    return (
        np.stack(audio).astype(np.float32),
        np.stack(video).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        groups,
        ids,
    )


def _pair_inputs(audio: np.ndarray, video: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    audio_n = (audio - norm["audio_mean"]) / norm["audio_std"]
    video_n = (video - norm["video_mean"]) / norm["video_std"]
    repeated_audio = np.repeat(audio_n[:, None, :], 4, axis=1)
    return np.concatenate([repeated_audio, video_n], axis=-1).astype(np.float32)


def _macro_category_metrics(
    probabilities: np.ndarray, targets: np.ndarray, categories: list[str]
) -> dict[str, Any]:
    from scripts.fit_avatar_decision_readout import compute_metrics

    if len(probabilities) != len(targets) or len(categories) != len(targets):
        raise ValueError("category metrics are not row-aligned")
    by_category = {}
    category_values = np.asarray(categories)
    for category in sorted(set(categories)):
        mask = category_values == category
        by_category[category] = compute_metrics(probabilities[mask], targets[mask])
    keys = ("accuracy", "nll", "brier", "ece_15", "mean_confidence")
    macro = {key: float(np.mean([row[key] for row in by_category.values()])) for key in keys}
    return {
        "category_count": len(by_category),
        "macro_over_categories": macro,
        "by_category": by_category,
    }


def _fit_cpu_macro_nll(
    torch: Any,
    train_x: np.ndarray,
    train_targets: np.ndarray,
    validation_x: np.ndarray,
    validation_targets: np.ndarray,
    validation_categories: list[str],
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    from scripts.fit_avatar_decision_readout import _metrics_for_logits, _model_class

    torch.manual_seed(SEED)
    model_type = _model_class(torch, train_x.shape[-1])
    model = model_type().to(torch.device("cpu"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_values = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_targets)
    validation_values = torch.from_numpy(validation_x)
    validation_labels = torch.from_numpy(validation_targets)
    rng = np.random.default_rng(SEED)
    history = []
    best_macro_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    last_state = None
    started = time.perf_counter()
    for epoch in range(1, 51):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_targets))
        loss_sum = 0.0
        correct = 0
        seen = 0
        gradient_norms = []
        for first in range(0, len(order), 128):
            indices = order[first : first + 128]
            batch_x = train_values[indices]
            batch_y = train_labels[indices]
            logits = model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite AVATAR transfer gradient norm")
            gradient_norms.append(float(gradient_norm))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(dim=1) == batch_y).sum())
            seen += len(indices)
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_values)
            validation_metrics, validation_probabilities = _metrics_for_logits(
                validation_logits, validation_labels
            )
        macro = _macro_category_metrics(
            validation_probabilities, validation_targets, validation_categories
        )
        macro_nll = macro["macro_over_categories"]["nll"]
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": loss_sum / seen,
                "train_accuracy": correct / seen,
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "max_gradient_norm": float(np.max(gradient_norms)),
                "inner_validation_sample_metrics": validation_metrics,
                "inner_validation_macro_metrics": macro["macro_over_categories"],
                "inner_validation_category_count": macro["category_count"],
                "selection_macro_nll": macro_nll,
                "epoch_wall_seconds": time.perf_counter() - epoch_started,
            }
        )
        last_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        if macro_nll < best_macro_nll - 0.001:
            best_macro_nll = macro_nll
            best_epoch = epoch
            best_state = last_state
            stale_epochs = 0
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(best_state, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= 6:
            break
    if best_state is None or last_state is None:
        raise RuntimeError("macro-NLL selector did not save checkpoints")
    last_path = checkpoint_path.with_name("last-state.pt")
    temporary = last_path.with_suffix(last_path.suffix + ".tmp")
    torch.save(last_state, temporary)
    temporary.replace(last_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        best_logits = model(validation_values)
        _, best_probabilities = _metrics_for_logits(best_logits, validation_labels)
    return {
        "model": model,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_macro_validation_nll": best_macro_nll,
        "selected_inner_validation_metrics": _macro_category_metrics(
            best_probabilities, validation_targets, validation_categories
        ),
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "train_input": args.train_input.resolve(strict=True),
        "train_sidecar": args.train_sidecar.resolve(strict=True),
        "audio_cache": args.audio_cache.resolve(strict=True),
        "video_cache": args.video_cache.resolve(strict=True),
    }
    expected = {
        "train_input": "0139f4f59a685c3f83890156b59c42d12aca9db54b22ee5c0a9c70b6ec7d60ef",
        "train_sidecar": "90adbf04450e0854fdb379dbcceb6aaa73790b205c08f1cf6c2cb786b6710a57",
        "audio_cache": "5736e2c7e094a93aea81fa13f517799292e790197dda6f6282140e3882636a58",
        "video_cache": "6ab47dcc0f9c8fe691425fa45a6f1394c8520677829a419c51b93a45da068b6e",
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != expected:
        raise ValueError(f"frozen TRAIN/cache hash mismatch: {hashes}")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite category-transfer output: {output}")

    source_rows = _read_jsonl(paths["train_input"])
    source_sidecars = _read_jsonl(paths["train_sidecar"])
    queries, videos = _query_and_video_pools(source_rows, source_sidecars)
    audio_keys = {str(row["asset_key"]) for row in queries}
    video_keys = {str(row["asset_key"]) for row in videos}
    # Only IDs referenced by the pinned TRAIN JSONL are selected from the combined cache.
    audio_features, audio_cache = _load_train_features(
        paths["audio_cache"], audio_keys, kind="audio", dimension=384
    )
    video_features, video_cache = _load_train_features(
        paths["video_cache"], video_keys, kind="video", dimension=768
    )
    with np.load(paths["audio_cache"], allow_pickle=False) as archive:
        all_audio_keys = [str(value) for value in archive["asset_keys"].tolist()]
        audio_index = {key: index for index, key in enumerate(all_audio_keys)}
        selected_audio_keys = sorted(audio_keys)
        audio_positions = [audio_index[key] for key in selected_audio_keys]
        centers = np.asarray(archive["centers_seconds"][audio_positions])
        padding = np.asarray(archive["declared_padding_samples"][audio_positions], dtype=np.int64)
    query_by_key = {row["asset_key"]: row for row in queries}
    for key, center, pad in zip(selected_audio_keys, centers, padding, strict=True):
        query = query_by_key[key]
        expected_pad = (
            round(float(query["left_padding_seconds"]) * 32_000),
            round(float(query["right_padding_seconds"]) * 32_000),
        )
        if abs(float(center) - float(query["timestamp_seconds"])) > 1e-9:
            raise ValueError(f"TRAIN audio timestamp differs from frozen input: {key}")
        if tuple(int(value) for value in pad) != expected_pad:
            raise ValueError(f"TRAIN audio padding differs from frozen input: {key}")
    with np.load(paths["video_cache"], allow_pickle=False) as archive:
        all_video_keys = [str(value) for value in archive["asset_keys"].tolist()]
        video_index = {key: index for index, key in enumerate(all_video_keys)}
        selected_video_keys = sorted(video_keys)
        video_positions = [video_index[key] for key in selected_video_keys]
        timestamps = np.asarray(archive["timestamps"][video_positions])
    video_by_key = {row["asset_key"]: row for row in videos}
    for key, timestamp in zip(selected_video_keys, timestamps, strict=True):
        if abs(float(timestamp) - float(video_by_key[key]["timestamp_seconds"])) > 1e-9:
            raise ValueError(f"TRAIN video timestamp differs from frozen input: {key}")
    partition = build_transfer_partition(queries, videos)
    seen = set(partition["seen_categories"])
    held = set(partition["held_categories"])
    choices = {
        "fit": build_choices(
            queries,
            videos,
            split="fit",
            allowed_groups=partition["fit_groups"],
            allowed_categories=seen,
        ),
        "inner_validation": build_choices(
            queries,
            videos,
            split="inner_validation",
            allowed_groups=partition["validation_groups"],
            allowed_categories=seen,
        ),
        "held_category_test": build_choices(
            queries,
            videos,
            split="held_category_test",
            allowed_groups=partition["held_test_groups"],
            allowed_categories=held,
        ),
    }
    integrity = validate_choice_splits(choices)
    fit_categories = {category for row in choices["fit"] for category in row["option_categories"]}
    if fit_categories & held:
        raise ValueError("held categories leaked into fit option labels")
    validation_categories = {
        category for row in choices["inner_validation"] for category in row["option_categories"]
    }
    if validation_categories & held:
        raise ValueError("held categories leaked into inner validation")
    if set(partition["held_test_groups"]) & (
        set(partition["fit_groups"]) | set(partition["validation_groups"])
    ):
        raise ValueError("source group partition overlap")
    if len(choices["fit"]) < 3_000 or len(choices["inner_validation"]) < 500:
        raise ValueError("TRAIN-only choice coverage below preregistered minimum")
    if len(choices["held_category_test"]) < 1_000:
        raise ValueError("held-category test coverage below preregistered minimum")
    for category in held:
        support = sum(row["category"] == category for row in choices["held_category_test"])
        groups = {
            row["audio_source_group"]
            for row in choices["held_category_test"]
            if row["category"] == category
        }
        if support == 0 or len(groups) < 4:
            raise ValueError(f"held category lacks minimum query support: {category}")

    train_audio, train_video, train_targets, train_groups, train_ids = _arrays(
        choices["fit"], audio_features, video_features
    )
    val_audio, val_video, val_targets, val_groups, val_ids = _arrays(
        choices["inner_validation"], audio_features, video_features
    )
    test_audio, test_video, test_targets, test_groups, test_ids = _arrays(
        choices["held_category_test"], audio_features, video_features
    )
    norm = {
        "audio_mean": train_audio.mean(axis=0),
        "audio_std": train_audio.std(axis=0).clip(1e-5),
        "video_mean": train_video.reshape(-1, 768).mean(axis=0),
        "video_std": train_video.reshape(-1, 768).std(axis=0).clip(1e-5),
    }
    train_x = _pair_inputs(train_audio, train_video, norm)
    val_x = _pair_inputs(val_audio, val_video, norm)
    test_x = _pair_inputs(test_audio, test_video, norm)
    output.mkdir(parents=True)
    (output / "joint").mkdir()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("category-transfer experiment must remain CPU-only")
    from scripts.fit_avatar_decision_readout import _metrics_for_logits, compute_metrics

    torch_threads = max(1, min(2, os.cpu_count() or 1))
    torch.set_num_threads(torch_threads)
    start_utc = datetime.now(UTC).isoformat()
    effective_config = {
        "experiment_id": "avatar-category-transfer-v1",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "seed": SEED,
        "held_category_count": HELD_CATEGORY_COUNT,
        "validation_fraction": VALIDATION_FRACTION,
        "held_categories": partition["held_categories"],
        "seen_categories": partition["seen_categories"],
        "train_input_sha256": hashes["train_input"],
        "train_sidecar_sha256": hashes["train_sidecar"],
        "audio_feature_cache_sha256": hashes["audio_cache"],
        "video_feature_cache_sha256": hashes["video_cache"],
        "model": "per-option 1152->64 GELU->1 MLP; 73,857 parameters",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 128,
            "max_epochs": 50,
            "early_stopping_patience": 6,
            "selector": "macro-category inner-validation NLL",
            "min_delta": 0.001,
        },
        "runtime": {
            "device": "CPU",
            "cuda_visible_devices": "-1",
            "torch_num_threads": torch_threads,
        },
        "normalization": (
            "mean/std fit on fit-only audio queries and video candidate features; std floor 1e-5"
        ),
        "evaluation": (
            "held-category TRAIN subset scored once after best-checkpoint selection; "
            "no original validation or sealed audit"
        ),
    }
    config_path = output / "effective-config.json"
    config_path.write_text(
        json.dumps(effective_config, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    fit = _fit_cpu_macro_nll(
        torch,
        train_x,
        train_targets,
        val_x,
        val_targets,
        [row["category"] for row in choices["inner_validation"]],
        checkpoint_path=output / "joint" / "best-state.pt",
    )
    if fit["parameter_count"] != 73_857:
        raise AssertionError(f"unexpected scorer parameter count: {fit['parameter_count']}")
    model = fit["model"]
    test_started = time.perf_counter()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(test_x))
    test_metrics, test_probabilities = _metrics_for_logits(
        test_logits, torch.from_numpy(test_targets)
    )
    test_eval_seconds = time.perf_counter() - test_started
    category_metrics = {}
    for category in sorted(held):
        mask = np.asarray([row["category"] == category for row in choices["held_category_test"]])
        category_metrics[category] = {
            **compute_metrics(test_probabilities[mask], test_targets[mask]),
            "query_source_groups": len(
                {group for group, keep in zip(test_groups, mask, strict=True) if keep}
            ),
        }
    predictions = []
    for index, row in enumerate(choices["held_category_test"]):
        probs = test_probabilities[index].tolist()
        predictions.append(
            {
                "query_id": row["query_id"],
                "audio_source_group": row["audio_source_group"],
                "category": row["category"],
                "target_index": int(test_targets[index]),
                "prediction_index": int(np.argmax(probs)),
                "probabilities": probs,
                "option_categories": row["option_categories"],
                "option_source_groups": row["video_source_groups"],
                "option_asset_keys": row["video_asset_keys"],
            }
        )
    with (output / "held-category-predictions.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "seed": SEED,
        "held_categories": partition["held_categories"],
        "seen_categories": partition["seen_categories"],
        "splits": {
            name: {
                "query_ids": [row["query_id"] for row in rows],
                "query_groups": sorted({row["audio_source_group"] for row in rows}),
            }
            for name, rows in choices.items()
        },
    }
    manifest_path = output / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "COMPLETED_TRAIN_ONLY_HELD_CATEGORY_TRANSFER_DIAGNOSTIC",
        "run_started_utc": start_utc,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "python": sys.version,
        "platform": sys.platform,
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "CPU (CUDA_VISIBLE_DEVICES=-1; cuda_available=False verified before fit)",
        "command": [
            "python",
            str(Path(__file__).resolve()),
            "--train-input",
            str(paths["train_input"]),
            "--train-sidecar",
            str(paths["train_sidecar"]),
            "--audio-cache",
            str(paths["audio_cache"]),
            "--video-cache",
            str(paths["video_cache"]),
            "--output-dir",
            str(output),
        ],
        "seed": SEED,
        "experiment": {
            "category_holdout_rule": (
                "first 16 by SHA256(avatar-category-transfer-v1:17:<category>)"
            ),
            "source_group_rule": (
                "groups with any held category in TRAIN query/video metadata go to held test; "
                "remaining groups split 80/20 by SHA256(avatar-inner-groups-v1:17:<group>)"
            ),
            "candidate_pools": (
                "rebuilt independently within fit, inner-validation, and held-category-test pools"
            ),
            "model": ("per-option 1152->64 GELU->1 concatenation scorer; same AVATAR joint MLP"),
            "optimizer": (
                "AdamW(lr=1e-3, weight_decay=1e-4); batch 128; max 50 epochs; "
                "patience 6; macro validation NLL min_delta 0.001"
            ),
            "checkpoint_selection": (
                "inner TRAIN-only macro-category NLL; held-category test after selection"
            ),
            "held_data_usage": (
                "TRAIN rows/features only; no original validation labels or sealed audit"
            ),
            "feature_encoders": "frozen; no encoder calls or training",
            "limits": (
                "exploratory category-transfer diagnostic; not an external audit or product claim"
            ),
        },
        "input_hashes": hashes,
        "effective_config_sha256": _sha256(config_path),
        "feature_cache_selection": {
            "audio": audio_cache,
            "video": video_cache,
            "note": (
                "combined cache files were filtered to IDs referenced by pinned TRAIN input; "
                "no original validation labels or predictions were read"
            ),
        },
        "data": {
            "source_train_rows": len(source_rows),
            "unique_audio_queries": len(queries),
            "unique_candidate_video_assets": len(videos),
            "all_source_group_count": len(partition["group_categories"]),
            "held_categories": partition["held_categories"],
            "seen_categories": partition["seen_categories"],
            "fit_category_count": len(fit_categories),
            "inner_validation_category_count": len(validation_categories),
            "partition_integrity": integrity,
            "held_test_category_counts": dict(
                Counter(row["category"] for row in choices["held_category_test"])
            ),
            "fit_choice_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
            "inner_validation_choice_ids_sha256": hashlib.sha256(
                "\n".join(val_ids).encode()
            ).hexdigest(),
            "held_test_choice_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
            "split_manifest_sha256": _sha256(manifest_path),
        },
        "training": {
            "trainable_parameters": fit["parameter_count"],
            "best_epoch": fit["best_epoch"],
            "epochs_run": fit["epochs_run"],
            "best_inner_validation_macro_nll": fit["best_macro_validation_nll"],
            "inner_validation_metrics": fit["selected_inner_validation_metrics"],
            "fit_history": fit["history"],
            "fit_wall_seconds": fit["wall_seconds"],
            "optimizer_update_count": len(fit["history"]) * ((len(choices["fit"]) + 127) // 128),
            "peak_cuda_allocated_bytes": 0,
            "best_checkpoint_bytes": (output / "joint" / "best-state.pt").stat().st_size,
            "checkpoint_sha256": _sha256(output / "joint" / "best-state.pt"),
        },
        "held_category_test": {
            "metrics": test_metrics,
            "macro_category_metrics": _macro_category_metrics(
                test_probabilities,
                test_targets,
                [row["category"] for row in choices["held_category_test"]],
            )["macro_over_categories"],
            "per_category": category_metrics,
            "evaluation_seconds": test_eval_seconds,
            "accuracy_source_group_bootstrap_95": _cluster_bootstrap_accuracy(
                test_probabilities, test_targets, test_groups
            ),
            "predictions_sha256": _sha256(output / "held-category-predictions.jsonl"),
            "test_predictions_used_for_selection": False,
        },
        "parameter_accounting": {
            "audio_encoder": 1_961_239,
            "video_encoder": 114_365_184,
            "trainable_readout": 73_857,
            "this_av_route_total": 116_400_280,
            "note": "task-scoped AV route; not the complete four-modality product total",
        },
        "compute_cost": {"local_cpu_only": True, "cloud_cost_usd": 0},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["report_sha256"] = _sha256(output / "report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--train-sidecar", required=True, type=Path)
    parser.add_argument("--audio-cache", required=True, type=Path)
    parser.add_argument("--video-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
