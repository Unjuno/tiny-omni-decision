from __future__ import annotations

from pathlib import Path

from tiny_omni_decision.clevr4_quality import _bucket, build_clevr4_quality_splits
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import DatasetManifest

ROOT = Path(__file__).parents[1]


def test_clevr4_quality_split_keeps_images_atomic_and_reserves_fresh_assets() -> None:
    manifest = DatasetManifest.model_validate(
        load_structured_file(ROOT / "manifests" / "candidates" / "clevr4.yaml")
    )
    annotations: dict[str, dict[str, str | int]] = {}
    validation_targets = {key: set() for key in CLEVR4_TAXONOMIES}
    train_id = None
    reserve_id = None
    for index in range(200_000):
        asset_id = f"fresh-{index:06d}"
        bucket = _bucket(asset_id, 17)
        if bucket < 70 and train_id is None:
            train_id = asset_id
        elif bucket >= 85 and reserve_id is None:
            reserve_id = asset_id
        if 70 <= bucket < 85:
            missing = [
                taxonomy
                for taxonomy, values in CLEVR4_TAXONOMIES.items()
                if len(validation_targets[taxonomy]) < len(values)
            ]
            if missing:
                row: dict[str, str | int] = {"split": "train"}
                for taxonomy, values in CLEVR4_TAXONOMIES.items():
                    value_list = list(map(str, values))
                    value = next(
                        (
                            candidate
                            for candidate in value_list
                            if candidate not in validation_targets[taxonomy]
                        ),
                        value_list[index % len(value_list)],
                    )
                    row[taxonomy] = value
                    validation_targets[taxonomy].add(value)
                annotations[asset_id] = row
        if train_id and reserve_id and all(
            len(validation_targets[taxonomy]) == len(values)
            for taxonomy, values in CLEVR4_TAXONOMIES.items()
        ):
            break
    assert train_id is not None and reserve_id is not None
    annotations[train_id] = {
        "split": "train",
        **{key: str(values[0]) for key, values in CLEVR4_TAXONOMIES.items()},
    }
    annotations[reserve_id] = {
        "split": "train",
        **{key: str(values[0]) for key, values in CLEVR4_TAXONOMIES.items()},
    }
    annotations["seen-before"] = {
        "split": "train",
        **{key: str(values[0]) for key, values in CLEVR4_TAXONOMIES.items()},
    }
    annotations["official-validation"] = {
        "split": "validation",
        **{key: str(values[0]) for key, values in CLEVR4_TAXONOMIES.items()},
    }

    training, validation, report = build_clevr4_quality_splits(
        annotations,
        previously_seen_image_ids={"seen-before"},
        manifest=manifest,
        seed=17,
    )
    train_images = {item.task_group_id for item in training}
    validation_images = {item.task_group_id for item in validation}
    assert len(training) == 4 * len(train_images)
    assert len(validation) == 4 * len(validation_images)
    assert train_images.isdisjoint(validation_images)
    assert report["train_validation_image_overlap"] == 0
    assert report["fresh_official_train_images"] == (
        len(train_images) + len(validation_images) + report["unmaterialized_reserve_images"]
    )
    assert all(report["class_counts_by_taxonomy"]["validation"][name] for name in CLEVR4_TAXONOMIES)
