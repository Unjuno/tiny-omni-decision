from copy import deepcopy

import pytest

from scripts.download_habit_pilot import REVISION, validate_inventory


def _inventory():
    episode = 12
    chunk = episode // 1000
    return {
        "dataset": "configinc/HABIT",
        "revision": REVISION,
        "payload_downloaded": False,
        "sealed_audit_accessed": False,
        "camera_view": "observation.images.front_view",
        "file_count": 2,
        "episode_count": 1,
        "total_bytes": 3,
        "items": [
            {
                "split": "train",
                "episode_index": episode,
                "kind": "parquet",
                "path": f"full/data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
                "size": 1,
                "sha256": "a" * 64,
            },
            {
                "split": "train",
                "episode_index": episode,
                "kind": "video",
                "path": (
                    f"full/videos/chunk-{chunk:03d}/observation.images.front_view/"
                    f"episode_{episode:06d}.mp4"
                ),
                "size": 2,
                "sha256": "b" * 64,
            },
        ],
    }


def test_valid_inventory_contains_one_train_parquet_and_front_video():
    items = validate_inventory(_inventory())

    assert [item["kind"] for item in items] == ["parquet", "video"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["items"][0].update(split="sealed_audit"), "train and validation"),
        (lambda data: data["items"].pop(), "exactly one video and Parquet"),
        (lambda data: data["items"][1].update(path="full/videos/other/episode_000012.mp4"),
         "approved camera"),
        (lambda data: data.update(revision="mutable-main"), "identity"),
        (lambda data: data["items"][0].update(sha256="not-a-hash"), "SHA-256"),
    ],
)
def test_invalid_inventory_is_rejected(mutate, message):
    inventory = deepcopy(_inventory())
    mutate(inventory)

    with pytest.raises(ValueError, match=message):
        validate_inventory(inventory)


def test_same_episode_cannot_be_assigned_to_both_splits():
    inventory = _inventory()
    second = deepcopy(inventory["items"][0])
    second["split"] = "validation"
    inventory["items"].append(second)
    inventory["file_count"] += 1
    inventory["episode_count"] += 1
    inventory["total_bytes"] += second["size"]

    with pytest.raises(ValueError, match="more than one split"):
        validate_inventory(inventory)
