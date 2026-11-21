import hashlib
import json

import numpy as np
import pytest

from scripts.cv_tallyqa_vg_train_groups import (
    _read_train_image_features,
    assign_group_folds,
    group_keys,
    train_validation_indices,
)
from scripts.make_tallyqa_vg_train_only_manifest import make_train_only_manifest
from scripts.verify_tallyqa_vg_train_group_cv import (
    verify_artifact_index,
    verify_oof_identity,
)


def _row(question_id, image_id, question, answer):
    answers = ["0", "1", "2", "3"]
    target = answers.index(str(answer))
    return {
        "question_id": question_id,
        "image_id": image_id,
        "question": question,
        "target": target,
        "options": [{"id": value} for value in answers],
    }


def test_group_keys_connect_image_and_normalized_question_transitively():
    rows = [
        _row("q1", "im1", "How many cats?", "0"),
        _row("q2", "im2", "HOW MANY CATS", "1"),
        _row("q3", "im2", "How many dogs?", "2"),
        _row("q4", "im4", "How many birds?", "3"),
    ]

    keys = group_keys(rows)

    assert keys[0] == keys[1] == keys[2]
    assert keys[3] != keys[0]


def test_outer_group_folds_are_deterministic_and_never_split_groups():
    rows = [
        _row(f"q{i}", f"im{i}", f"How many type {i}?", str(i % 4))
        for i in range(20)
    ]
    keys = group_keys(rows)

    first = assign_group_folds(rows, keys, n_folds=5, seed=17)
    second = assign_group_folds(rows, keys, n_folds=5, seed=17)

    assert first == second
    assert set(first.values()) == set(range(5))
    assert all(len({first[key] for key in keys if key == group}) == 1 for group in first)
    relabeled = [dict(row, target=(row["target"] + 1) % 4) for row in rows]
    assert assign_group_folds(relabeled, keys, n_folds=5, seed=17) == first


def test_inner_stop_groups_are_disjoint_from_fit_groups():
    rows = [
        _row(f"q{i}", f"im{i}", f"Question {i}", str(i % 4))
        for i in range(25)
    ]
    keys = group_keys(rows)

    fit_indices, stop_indices, _ = train_validation_indices(rows, keys, seed=1026)

    assert set(fit_indices).isdisjoint(stop_indices)
    assert {keys[i] for i in fit_indices}.isdisjoint({keys[i] for i in stop_indices})
    assert sorted(fit_indices + stop_indices) == list(range(len(rows)))


def test_train_only_manifest_excludes_all_heldout_partitions(tmp_path):
    source = tmp_path / "task.json"
    source.write_text(
        '{"train":[{"question_id":"q1"}],"validation":[{"question_id":"v1"}],'
        '"test":[{"question_id":"t1"}]}',
        encoding="utf-8",
    )

    result = make_train_only_manifest(source)

    assert result["split_scope"] == "train_only"
    assert result["train"] == [{"question_id": "q1"}]
    assert "validation" not in result
    assert "test" not in result


def test_train_image_cache_accepts_repeated_questions_for_same_image(tmp_path):
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        train_ids=np.asarray(["im1", "im2"]),
        train_media_sha256=np.asarray(["hash1", "hash2"]),
        train_features=np.zeros((2, 768), dtype=np.float32),
    )
    rows = [
        {"image_id": "im1", "media_sha256": "hash1"},
        {"image_id": "im1", "media_sha256": "hash1"},
        {"image_id": "im2", "media_sha256": "hash2"},
    ]

    features = _read_train_image_features(cache, rows)

    assert set(features) == {"im1", "im2"}
    assert tuple(features["im1"].shape) == (768,)


def test_artifact_index_verifies_hashes_and_exact_coverage(tmp_path):
    payload = b"checkpoint"
    (tmp_path / "checkpoint.bin").write_bytes(payload)
    (tmp_path / "artifact-index.json").write_text(json.dumps({
        "artifact_count": 1,
        "total_bytes": len(payload),
        "artifacts": [{
            "path": "checkpoint.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }), encoding="utf-8")

    result = verify_artifact_index(tmp_path)

    assert result["artifact_count"] == 1
    assert result["total_bytes"] == len(payload)


def test_artifact_index_rejects_modified_checkpoint(tmp_path):
    (tmp_path / "checkpoint.bin").write_bytes(b"changed")
    (tmp_path / "artifact-index.json").write_text(json.dumps({
        "artifact_count": 1,
        "total_bytes": 10,
        "artifacts": [{"path": "checkpoint.bin", "bytes": 10, "sha256": "0" * 64}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_artifact_index(tmp_path)


def test_oof_identity_rejects_fold_corruption():
    rows = [
        dict(_row("q1", "im1", "How many cats?", "0"), media_sha256="a"),
        dict(_row("q2", "im2", "How many dogs?", "1"), media_sha256="b"),
        dict(_row("q3", "im3", "How many birds?", "2"), media_sha256="c"),
        dict(_row("q4", "im4", "How many horses?", "3"), media_sha256="d"),
        dict(_row("q5", "im5", "How many fish?", "0"), media_sha256="e"),
    ]
    group_assignment = assign_group_folds(rows, group_keys(rows), n_folds=5, seed=17)
    keys = group_keys(rows)
    predictions = []
    for row, key in zip(rows, keys, strict=True):
        prediction = {
            "question_id": row["question_id"], "target": row["options"][row["target"]]["id"],
            "prediction": row["options"][0]["id"], "options": [x["id"] for x in row["options"]],
            "probabilities": [0.25] * 4,
        }
        predictions.append({
            "question_id": row["question_id"], "image_id": row["image_id"], "group_key": key,
            "target_option_index": row["target"], "fold": group_assignment[key],
            "arms": {arm: prediction for arm in ("joint", "question_only", "image_zero")},
        })

    assert verify_oof_identity(rows, predictions)["unique_question_ids"] == 5
    predictions[0]["fold"] = (predictions[0]["fold"] + 1) % 5
    with pytest.raises(ValueError, match="identity/split/target mismatch"):
        verify_oof_identity(rows, predictions)
