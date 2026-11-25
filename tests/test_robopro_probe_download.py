import pytest

from scripts.download_robopro_probe_sample import _selected


def _sample():
    return {
        "train": [
            {
                "prompt_sha256": "a",
                "episodes": [{"episode_id": value} for value in range(4)],
            }
        ],
        "development": [
            {
                "prompt_sha256": "b",
                "episodes": [{"episode_id": value} for value in range(10, 14)],
            }
        ],
    }


def test_download_selection_is_partitioned_and_complete():
    selected = _selected(_sample())

    assert len(selected) == 8
    assert len({episode_id for _, episode_id in selected}) == 8
    assert {split for split, _ in selected} == {"train", "development"}


def test_download_selection_rejects_cross_partition_episode_overlap():
    sample = _sample()
    sample["development"][0]["episodes"][0]["episode_id"] = 0

    with pytest.raises(ValueError, match="overlap across partitions"):
        _selected(sample)


def test_download_selection_rejects_repeated_prompt_groups():
    sample = _sample()
    sample["train"].append(sample["train"][0])

    with pytest.raises(ValueError, match="duplicate prompt group"):
        _selected(sample)
