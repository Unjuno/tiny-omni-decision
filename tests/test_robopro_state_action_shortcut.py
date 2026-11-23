from __future__ import annotations

import hashlib

import pytest

from scripts.audit_robopro_state_action_shortcut import (
    midranks,
    score_prompt_group,
    validate_frozen_split,
)


def _frozen_sample_and_split() -> tuple[dict, dict]:
    sample = {}
    split = {}
    for name, id_offset, prompt_offset in (
        ("train", 0, 0),
        ("development", 100, 16),
    ):
        groups = []
        prompts = []
        episode_ids = []
        for group_index in range(16):
            prompt = f"prompt-{prompt_offset + group_index}"
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            prompts.append(prompt)
            ids = list(range(id_offset + group_index * 4, id_offset + group_index * 4 + 4))
            episode_ids.extend(ids)
            groups.append(
                {
                    "prompt": prompt,
                    "prompt_sha256": prompt_hash,
                    "episodes": [{"episode_id": episode_id} for episode_id in ids],
                }
            )
        sample[name] = groups
        split[name] = {"prompt_groups": prompts, "episode_ids": episode_ids}
    return sample, split


def test_midranks_assigns_average_rank_to_ties() -> None:
    assert midranks([1.0, 1.0, 3.0, 4.0]) == [1.5, 1.5, 3.0, 4.0]


def test_state_action_rank_gives_fractional_credit_for_top_tie() -> None:
    episodes = [
        {
            "episode_id": index,
            "states": [[0.0, 0.0]],
            "actions": [[0.0, 1.0], [1.0, 0.0], [3.0, 3.0], [4.0, 4.0]][index : index + 1],
        }
        for index in range(4)
    ]
    result = score_prompt_group(
        episodes,
        dimensions=[0, 1],
        progress_points=[0.0],
    )
    assert len(result) == 4
    assert result[0]["winner_indices"] == [0, 1]
    assert result[0]["top_tie"]
    assert result[0]["fractional_credit"] == 0.5


def test_state_action_rank_handles_episodes_with_different_lengths() -> None:
    episodes = [
        {
            "episode_id": index,
            "states": [[float(index), 0.0]] * (index + 1),
            "actions": [[float(index), 0.0]] * (index + 1),
        }
        for index in range(4)
    ]
    result = score_prompt_group(episodes, dimensions=[0], progress_points=[0.5])
    assert len(result) == 4


def test_state_action_rank_rejects_non_four_way_groups() -> None:
    with pytest.raises(ValueError, match="exactly four episodes"):
        score_prompt_group([], dimensions=[0], progress_points=[0.5])


def test_frozen_sample_requires_disjoint_train_and_development() -> None:
    sample, split = _frozen_sample_and_split()
    sample["development"][0]["episodes"][0]["episode_id"] = 0
    split["development"]["episode_ids"][0] = 0

    with pytest.raises(ValueError, match="identities overlap"):
        validate_frozen_split(sample, split)


def test_frozen_sample_rejects_prompt_hash_or_membership_mismatch() -> None:
    sample, split = _frozen_sample_and_split()
    sample["development"][0]["prompt"] = sample["train"][0]["prompt"]

    with pytest.raises(ValueError, match="prompt hash mismatch"):
        validate_frozen_split(sample, split)
