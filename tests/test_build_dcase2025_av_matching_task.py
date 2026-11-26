import pytest

pytest.importorskip("numpy")

from scripts.build_dcase2025_av_matching_task import build_examples  # noqa: E402


def _rows():
    return [
        {
            "sample_id": f"{room}-{event}-{parent}-{sample}",
            "room_id": room,
            "event_class": event,
            "parent_recording": f"{room}-{parent}",
            "target_onscreen": sample % 2,
        }
        for room in ("r1", "r2", "r3")
        for event in (0, 1)
        for parent in range(4)
        for sample in range(parent + 1)
    ]


def test_matching_task_is_deterministic_balanced_and_uses_hard_grouped_negatives():
    first, cells = build_examples(_rows(), seed=17)
    replay, replay_cells = build_examples(_rows(), seed=17)

    assert first == replay
    assert cells == replay_cells
    assert len(first) == 24
    assert len(cells) == 6
    assert {
        sum(example["target_index"] == position for example in first) for position in range(4)
    } == {6}
    for example in first:
        options = example["options"]
        assert len(options) == 4
        assert len({option["parent_recording"] for option in options}) == 4
        assert all(
            option["parent_recording"] != example["query_parent"]
            for i, option in enumerate(options)
            if i != example["target_index"]
        )
        assert options[example["target_index"]]["parent_recording"] == example["query_parent"]
        assert {example["room_id"]} == {
            option["parent_recording"].split("-")[0] for option in options
        }
        assert len({option["sample_id"] for option in options}) == 4


def test_matching_task_excludes_cells_with_fewer_than_four_parents():
    rows = _rows() + [
        {
            "sample_id": "rare-sample",
            "room_id": "r1",
            "event_class": 2,
            "parent_recording": "rare-parent",
        }
    ]
    examples, cells = build_examples(rows, seed=17)

    assert len(examples) == 24
    assert all(event != 2 for _, event in cells)
