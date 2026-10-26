from scripts.audit_robomnist_task_dependencies import dependency_accounting


def row(video, first, second):
    return {
        "source_refs": {
            "video": {"group_id": video},
            "audio_candidates": [{"group_id": first}, {"group_id": second}],
        }
    }


def test_reused_candidates_merge_transitively_and_order_duplicates_add_no_units():
    rows = [row("a", "a", "b"), row("b", "b", "c"), row("a", "b", "a"), row("d", "d", "e")]
    result = dependency_accounting(rows)
    assert result["rows"] == 4
    assert result["unique_repetitions"] == 5
    assert result["asset_sharing_connected_components"] == 2
    assert result["component_sizes"] == {2: 1, 3: 1}
    assert not result["independent_sessions_verified"]


def test_disjoint_questions_remain_separate():
    result = dependency_accounting([row("a", "a", "b"), row("c", "c", "d")])
    assert result["asset_sharing_connected_components"] == 2
