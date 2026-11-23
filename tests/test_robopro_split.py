from tiny_omni_decision.robopro_split import build_prompt_group_split


def _fixture():
    tasks = [
        {"task_index": 0, "task": "move the cup"},
        {"task_index": 1, "task": "place the cup"},
        {"task_index": 2, "task": "close the drawer"},
        {"task_index": 3, "task": "open the drawer"},
        {"task_index": 4, "task": "lift the plate"},
    ]
    task_by_id = {task["task_index"]: task["task"] for task in tasks}
    episodes = []
    for episode_id in range(16000):
        task_id = episode_id % len(tasks)
        if episode_id == 15999:
            task_id = 12000 % len(tasks)
        length = 20 if episode_id in (12000, 15999) else 10 + episode_id
        episodes.append(
            {
                "episode_index": episode_id,
                "length": length,
                "tasks": [task_by_id[task_id]],
            }
        )
    info = {
        "total_episodes": 16000,
        "total_tasks": len(tasks),
        "splits": {"train": "0:16000"},
    }
    return info, tasks, episodes


def test_prompt_groups_are_disjoint_and_duplicate_is_excluded():
    info, tasks, episodes = _fixture()
    result = build_prompt_group_split(info, tasks, episodes, seed=17)

    train = result["train"]
    development = result["development"]
    assert set(train["episode_ids"]).isdisjoint(development["episode_ids"])
    assert set(train["prompt_groups"]).isdisjoint(development["prompt_groups"])
    assert 15999 not in train["episode_ids"]
    assert 15999 not in development["episode_ids"]
    assert 12000 in train["episode_ids"] or 12000 in development["episode_ids"]
    assert result["counts"]["included_episodes"] == len(episodes) - 1


def test_seed_is_deterministic_and_changes_group_assignment():
    info, tasks, episodes = _fixture()
    first = build_prompt_group_split(info, tasks, episodes, seed=17)
    replay = build_prompt_group_split(info, tasks, episodes, seed=17)
    other = build_prompt_group_split(info, tasks, episodes, seed=23)

    assert first == replay
    assert first["development"]["prompt_groups"] != other["development"]["prompt_groups"]


def test_rejects_unknown_or_multi_prompt_episode():
    info, tasks, episodes = _fixture()
    episodes[0]["tasks"] = ["move the cup", "place the cup"]
    try:
        build_prompt_group_split(info, tasks, episodes)
    except ValueError as exc:
        assert "exactly one known task prompt" in str(exc)
    else:
        raise AssertionError("invalid episode prompt mapping was accepted")
