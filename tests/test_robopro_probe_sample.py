import pytest

from tiny_omni_decision.robopro_probe_sample import build_probe_sample


def _inputs():
    episodes = []
    train_ids = []
    development_ids = []
    for split, prompts, target in (
        ("train", ["train-a", "train-b", "train-c"], train_ids),
        ("development", ["dev-a", "dev-b", "dev-c"], development_ids),
    ):
        for prompt_index, prompt in enumerate(prompts):
            for offset in range(4):
                episode_id = (0 if split == "train" else 100) + prompt_index * 10 + offset
                target.append(episode_id)
                episodes.append(
                    {
                        "episode_index": episode_id,
                        "tasks": [prompt],
                        "length": 101 + offset,
                    }
                )
    return episodes, {
        "dataset": "test/RoboPro",
        "revision": "a" * 40,
        "seed": 17,
        "train": {"episode_ids": train_ids},
        "development": {"episode_ids": development_ids},
    }


def test_sample_is_deterministic_grouped_and_disjoint():
    episodes, manifest = _inputs()
    first = build_probe_sample(episodes, manifest, prompts_per_split=2)
    second = build_probe_sample(list(reversed(episodes)), manifest, prompts_per_split=2)

    assert first == second
    assert first["train_episode_count"] == 8
    assert first["development_episode_count"] == 8
    train_prompts = {group["prompt"] for group in first["train"]}
    development_prompts = {group["prompt"] for group in first["development"]}
    assert train_prompts.isdisjoint(development_prompts)
    assert all(len(group["episodes"]) == 4 for group in first["train"])
    assert all(len(group["episodes"]) == 4 for group in first["development"])


def test_exclusion_removes_the_entire_prompt_group():
    episodes, manifest = _inputs()
    first = build_probe_sample(episodes, manifest, prompts_per_split=3)
    excluded_id = first["train"][0]["episodes"][0]["episode_id"]
    prompt = first["train"][0]["prompt"]

    second = build_probe_sample(
        episodes,
        manifest,
        prompts_per_split=2,
        excluded_episode_ids={excluded_id},
    )

    assert prompt not in {group["prompt"] for group in second["train"]}
    selected_ids = {
        episode["episode_id"]
        for split in ("train", "development")
        for group in second[split]
        for episode in group["episodes"]
    }
    assert excluded_id not in selected_ids


def test_rejects_seed_mismatch_insufficient_groups_and_malformed_prompt():
    episodes, manifest = _inputs()
    with pytest.raises(ValueError, match="seed must match"):
        build_probe_sample(episodes, manifest, seed=13)
    with pytest.raises(ValueError, match="only 3 eligible prompts"):
        build_probe_sample(episodes, manifest, prompts_per_split=4)
    episodes[0]["tasks"] = ["train-a", "other"]
    with pytest.raises(ValueError, match="invalid task prompt"):
        build_probe_sample(episodes, manifest, prompts_per_split=2)
