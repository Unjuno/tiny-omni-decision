from scripts.prepare_cater_temporal_decisions import _object_option, _query_row


def test_query_keeps_every_option_and_correct_target() -> None:
    scene = {
        "objects": [
            {
                "shape": "cube",
                "color": "red",
                "material": "rubber",
                "locations": {"0": [0, 0, 0], "75": [2, 0, 0]},
            },
            {
                "shape": "sphere",
                "color": "blue",
                "material": "metal",
                "locations": {"0": [0, 1, 0], "75": [0.5, 1, 0]},
            },
            {
                "shape": "cone",
                "color": "green",
                "material": "rubber",
                "locations": {"0": [0, 2, 0], "75": [0, 2, 0]},
            },
        ]
    }

    row = _query_row(17, "train", scene, (0, 75))
    assert row["id"] == "cater-000017-disp-000-075"
    assert row["group_id"] == "cater-video-000017"
    assert row["num_options"] == 3
    assert len(row["options"]) == 3
    assert row["target_option_id"] == _object_option(scene["objects"][0])["id"]


def test_target_is_independent_of_option_order() -> None:
    objects = [
        {
            "shape": "cube",
            "color": "red",
            "material": "rubber",
            "locations": {"0": [0, 0, 0], "75": [2, 0, 0]},
        },
        {
            "shape": "sphere",
            "color": "blue",
            "material": "metal",
            "locations": {"0": [0, 1, 0], "75": [0.5, 1, 0]},
        },
    ]

    a = _query_row(17, "train", {"objects": objects}, (0, 75))
    b = _query_row(17, "train", {"objects": list(reversed(objects))}, (0, 75))
    assert a["target_option_id"] == b["target_option_id"]
    assert a["options"] == b["options"]
