import hashlib
import json

import numpy as np
import pytest

from scripts.run_intphys_data_coverage_comparison import (
    _unique_group_representative_indices,
    deterministic_subset_indices,
    fit_fixed_update_budget,
    run_comparison,
)


def test_deterministic_subset_is_seeded_and_independent_of_source_order():
    sample_ids = [f"train.{shard}/{index:05d}" for shard in range(1, 4) for index in range(20)]
    selected = deterministic_subset_indices(sample_ids, 17, seed=17)
    selected_ids = [sample_ids[index] for index in selected]

    assert selected == deterministic_subset_indices(sample_ids, 17, seed=17)
    assert len(set(selected_ids)) == 17
    permuted = list(reversed(sample_ids))
    permuted_selected = [
        permuted[index]
        for index in deterministic_subset_indices(permuted, 17, seed=17)
    ]
    assert set(selected_ids) == set(permuted_selected)
    assert selected_ids != sample_ids[:17]


@pytest.mark.parametrize("count", [0, 4])
def test_deterministic_subset_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="subset count"):
        deterministic_subset_indices(["a", "b", "c"], count, seed=17)


def test_unique_group_representatives_are_stable_and_choose_one_per_group():
    ids = ["z-scene-a", "b-scene-a", "c-scene-b", "a-scene-b"]
    groups = ["scene-a", "scene-a", "scene-b", "scene-b"]

    selected = _unique_group_representative_indices(ids, groups)
    assert [ids[index] for index in selected] == ["a-scene-b", "b-scene-a"]
    permutation = [2, 0, 3, 1]
    permuted_ids = [ids[index] for index in permutation]
    permuted_groups = [groups[index] for index in permutation]
    selected_after_permutation = _unique_group_representative_indices(
        permuted_ids, permuted_groups
    )
    assert {permuted_ids[index] for index in selected_after_permutation} == {
        "a-scene-b",
        "b-scene-a",
    }


def test_fixed_update_fit_uses_exact_budget_and_finite_fixed_validation():
    generator = np.random.default_rng(17)
    train = generator.normal(size=(13, 4, 3)).astype(np.float32)
    validation = generator.normal(size=(7, 4, 3)).astype(np.float32)
    monitor = train[:5]

    best, terminal, history, best_step, best_loss = fit_fixed_update_budget(
        train,
        validation,
        monitor,
        seed=17,
        update_budget=5,
        eval_interval=2,
        batch_size=4,
    )

    assert best and terminal
    assert len(history) == 3
    assert [row["optimizer_update"] for row in history] == [2.0, 4.0, 5.0]
    assert history[-1]["train_example_presentations"] == 20.0
    assert history[-1]["unique_train_examples_consumed"] == 13.0
    assert history[-1]["unique_underlying_groups_consumed"] == 13.0
    assert best_step in {2, 4, 5}
    assert np.isfinite(best_loss)
    assert all(np.isfinite(row["validation_cosine_loss"]) for row in history)


def test_end_to_end_comparison_binds_shared_split_and_reload_receipts(tmp_path):
    train_path = tmp_path / "train-features.npz"
    validation_path = tmp_path / "validation-features.npz"
    generator = np.random.default_rng(19)
    train_count = 20
    validation_count = 6
    train_ids = [f"train.{index // 5 + 1}/{index:05d}" for index in range(train_count)]
    validation_ids = [f"validation/{index:05d}" for index in range(validation_count)]
    train_features = generator.normal(size=(train_count, 8, 768)).astype(np.float32)
    validation_features = generator.normal(size=(validation_count, 8, 768)).astype(np.float32)

    def identities(prefix, groups):
        return np.asarray(
            [hashlib.sha256(f"{prefix}-{group}".encode()).hexdigest() for group in groups]
        )

    train_groups = [f"train-group-{index}" for index in range(train_count)]
    train_groups[-1] = train_groups[-2]
    validation_groups = [f"validation-group-{index}" for index in range(validation_count)]
    validation_groups[-1] = validation_groups[-2]

    np.savez_compressed(
        train_path,
        features=train_features,
        sample_ids=np.asarray(train_ids),
        group_ids=np.asarray(train_groups),
        status_identity_sha256=identities("train-status", train_groups),
        rgb_sequence_sha256=identities("train-rgb", train_groups),
        sampled_rgb_pixels_sha256=identities("train-sampled", train_groups),
    )
    np.savez_compressed(
        validation_path,
        features=validation_features,
        sample_ids=np.asarray(validation_ids),
        group_ids=np.asarray(validation_groups),
        status_identity_sha256=identities("val-status", validation_groups),
        rgb_sequence_sha256=identities("val-rgb", validation_groups),
        sampled_rgb_pixels_sha256=identities("val-sampled", validation_groups),
    )
    from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
    from scripts.extract_frozen_videoprism_features import SOURCE_COMMIT
    from scripts.train_intphys_next_feature_predictor import sha256_file

    merge_report = {
        "status": "complete_intphys_all_train_shards_globally_grouped",
        "required_shards": [1, 2, 3, 4],
        "frame_count": 8,
        "feature_shape_per_sample": [8, 768],
        "model_revision": MODEL_REVISION,
        "videoprism_source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "feature_archive_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
        },
        "cross_split_group_overlap": 0,
        "cross_split_status_identity_overlap": 0,
        "cross_split_rgb_sequence_identity_overlap": 0,
        "cross_split_sampled_rgb_input_overlap": 0,
    }
    (tmp_path / "merge-report.json").write_text(
        json.dumps(merge_report), encoding="utf-8"
    )
    output_dir = tmp_path / "comparison"

    report = run_comparison(
        train_path,
        validation_path,
        output_dir,
        seed=17,
        small_train_count=8,
        update_budget=4,
        eval_interval=2,
        batch_size=4,
    )

    assert report["dev_or_sealed_audit_accessed"] is False
    assert report["train_examples_global"] == train_count - 1
    assert report["train_rows_before_group_deduplication"] == train_count
    assert report["train_duplicate_identity_rows_excluded"] == 1
    assert report["validation_examples_global"] == validation_count - 1
    assert report["validation_duplicate_identity_rows_excluded"] == 1
    assert report["unique_assets"] == {"D-small-8": 8, "D-full": train_count - 1}
    for name in ("D-small-8", "D-full"):
        metadata_path = output_dir / name / "run-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["best_reload_validation_cosine_loss"] == pytest.approx(
            metadata["selected_validation_cosine_loss"], abs=1e-8
        )
        assert report["candidates"][name]["run_metadata_sha256"] == sha256_file(metadata_path)
        assert (output_dir / name / "terminal-next-feature-predictor.pt").is_file()

