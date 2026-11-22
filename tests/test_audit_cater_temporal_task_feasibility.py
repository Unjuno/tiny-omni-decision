from scripts.audit_cater_temporal_task_feasibility import (
    distinct_visual_descriptors,
    temporal_query_windows,
)


def _object(shape: str, color: str, material: str, positions: list[list[float]]) -> dict:
    return {
        "shape": shape,
        "color": color,
        "material": material,
        "locations": {str(index * 75): position for index, position in enumerate(positions)},
    }


def test_temporal_queries_require_visually_distinguishable_objects() -> None:
    scene = {
        "objects": [
            _object("cube", "red", "rubber", [[0, 0, 0], [1, 0, 0]]),
            _object("cube", "red", "rubber", [[0, 2, 0], [0, 2, 0]]),
        ]
    }

    assert not distinct_visual_descriptors(scene["objects"])
    assert temporal_query_windows(scene, windows=((0, 75),)) == []


def test_temporal_queries_require_clear_top_displacement() -> None:
    scene = {
        "objects": [
            _object("cube", "red", "rubber", [[0, 0, 0], [1, 0, 0]]),
            _object("sphere", "blue", "metal", [[0, 2, 0], [0.1, 2, 0]]),
            _object("cone", "green", "rubber", [[0, 4, 0], [0, 4, 0]]),
        ]
    }

    assert temporal_query_windows(scene, windows=((0, 75),)) == [(0, 75)]
    assert temporal_query_windows(
        scene, windows=((0, 75),), min_winner_margin=1.0
    ) == []


def test_temporal_queries_reject_static_interval() -> None:
    scene = {
        "objects": [
            _object("cube", "red", "rubber", [[0, 0, 0], [0, 0, 0]]),
            _object("sphere", "blue", "metal", [[0, 2, 0], [0, 2, 0]]),
        ]
    }

    assert temporal_query_windows(scene, windows=((0, 75),)) == []
