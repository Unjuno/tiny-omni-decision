"""Describe cross-split visual-neighbor similarity in a frozen pilot feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_episode_similarity(features_path: Path) -> dict[str, object]:
    features = np.load(features_path, allow_pickle=False)
    episodes = features["episode"].astype(np.int64)
    splits = features["split"]
    video_features = features["video"].astype(np.float64)
    episode_ids = np.unique(episodes)
    episode_vectors = []
    episode_splits = []
    for episode_id in episode_ids:
        rows = episodes == episode_id
        split_values = np.unique(splits[rows])
        if len(split_values) != 1:
            raise ValueError(f"episode {episode_id} occurs in multiple splits")
        if int(rows.sum()) != 4:
            raise ValueError(f"episode {episode_id} must have four fixed windows")
        vector = video_features[rows].mean(axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        episode_vectors.append(vector)
        episode_splits.append(str(split_values[0]))

    vectors = np.stack(episode_vectors)
    similarity = vectors @ vectors.T
    train_indices = [i for i, split in enumerate(episode_splits) if split == "train"]
    validation_indices = [i for i, split in enumerate(episode_splits) if split == "validation"]
    cross_pairs = []
    for validation_index in validation_indices:
        for train_index in train_indices:
            cross_pairs.append(
                {
                    "validation_episode": int(episode_ids[validation_index]),
                    "train_episode": int(episode_ids[train_index]),
                    "cosine_similarity": float(similarity[validation_index, train_index]),
                    "episode_index_distance": int(
                        abs(episode_ids[validation_index] - episode_ids[train_index])
                    ),
                }
            )
    cross_pairs.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    values = np.asarray([row["cosine_similarity"] for row in cross_pairs])
    return {
        "feature_file_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "encoder": "facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "aggregation": (
            "L2-normalize mean of four sampled eight-frame-mean DINO CLS vectors per episode"
        ),
        "episode_groups": int(len(episode_ids)),
        "train_episodes": len(train_indices),
        "validation_episodes": len(validation_indices),
        "cross_split_pair_count": len(cross_pairs),
        "cross_split_cosine": {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "pairs_at_or_above_threshold": {
                str(threshold): int(np.sum(values >= threshold))
                for threshold in (0.98, 0.99, 0.995)
            },
        },
        "top_20_cross_split_neighbors": cross_pairs[:20],
        "interpretation_limit": (
            "Exploratory embedding similarity only; not a scene identity, duplicate, "
            "or independence determination. Threshold counts are descriptive, not gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_episode_similarity(args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["cross_split_cosine"], indent=2))


if __name__ == "__main__":
    main()
