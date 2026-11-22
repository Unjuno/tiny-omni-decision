import pytest

from scripts.pretrained_reuse.analyze_haptile_train_preflight import (
    validate_train_receipt,
)


def _receipt(task="insert_peg", path=None, split_hash="split-sha"):
    return {
        "format": "haptile_train_only_sample_preflight_v1",
        "revision": "c8593da98d3facb7a9293842514f0ee4ad3f2318",
        "split_manifest_sha256": split_hash,
        "files": [{
            "task": task,
            "path": path or f"data/{task}/episode_000.fo.mcap",
        }],
    }


def _manifest():
    return {
        "development_splits": {
            "train": ["insert_peg"],
            "calibration": ["put_spoon"],
            "validation": ["fold_tshirt"],
        },
    }


def test_validate_train_receipt_accepts_only_pinned_train_files():
    items = validate_train_receipt(_receipt(), _manifest(), "split-sha")
    assert items[0]["task"] == "insert_peg"


def test_validate_train_receipt_rejects_incomplete_full_corpus():
    receipt = _receipt()
    receipt["format"] = "haptile_frozen_train_corpus_v1"
    receipt["conversion_revision"] = receipt.pop("revision")
    receipt["split_manifest_file_sha256"] = receipt.pop("split_manifest_sha256")
    receipt["episode_count"] = 1
    with pytest.raises(ValueError, match="1,113"):
        validate_train_receipt(receipt, _manifest(), "split-sha")


def test_validate_train_receipt_accepts_complete_full_corpus_paths():
    train = [f"train_{index:02d}" for index in range(24)]
    files = []
    for index, task in enumerate(train):
        count = 47 if index < 9 else 46
        files.extend({"path": f"data/{task}/episode_{row:03d}.fo.mcap"} for row in range(count))
    receipt = {
        "format": "haptile_frozen_train_corpus_v1",
        "conversion_revision": "c8593da98d3facb7a9293842514f0ee4ad3f2318",
        "split_manifest_file_sha256": "split-sha",
        "episode_count": 1113,
        "files": files,
    }
    manifest = {
        "development_splits": {
            "train": train,
            "calibration": ["calibration_task"],
            "validation": ["fold_tshirt"],
        },
    }
    assert len(validate_train_receipt(receipt, manifest, "split-sha")) == 1113


@pytest.mark.parametrize(
    ("receipt", "manifest", "split_hash"),
    [
        (_receipt(task="fold_tshirt"), _manifest(), "split-sha"),
        (_receipt(), _manifest(), "other-hash"),
        (_receipt(path="data/put_spoon/episode_000.fo.mcap"), _manifest(), "split-sha"),
        (
            _receipt(path="data/insert_peg/../fold_tshirt/episode_000.fo.mcap"),
            _manifest(),
            "split-sha",
        ),
    ],
)
def test_validate_train_receipt_rejects_heldout_or_unsafe_inputs(
    receipt, manifest, split_hash
):
    with pytest.raises(ValueError):
        validate_train_receipt(receipt, manifest, split_hash)
