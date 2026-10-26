from scripts.build_starss23_spatial_decision_corpus import build_window_targets


def test_binary_corpus_targets_are_grouped_by_event_and_candidate_sector():
    window = {
        "id": "recording:t0000-0005",
        "window_index": 7,
        "recording": "recording",
        "room": "room-a",
        "split": "train",
        "start_seconds": 0,
        "end_seconds": 5,
    }
    annotations = [
        [0, 0, 1, 0, 0, 100],
        [49, 1, 2, 90, 0, 100],
        [50, 8, 0, -90, 0, 200],
    ]

    rows = build_window_targets(window, annotations)

    assert len(rows) == 96
    assert rows[0]["event"] == "speech"
    assert rows[0]["sector"] == "front"
    assert rows[0]["options"] == ["no", "yes"]
    assert rows[0]["target"] == 1
    assert rows[2]["sector"] == "left"
    assert rows[2]["target"] == 1
    assert rows[7 * 8 + 6]["event"] == "music"
    assert rows[7 * 8 + 6]["target"] == 0
    assert {row["window_index"] for row in rows} == {7}
    assert {row["split"] for row in rows} == {"train"}
    assert len({row["sample_id"] for row in rows}) == 96
