import importlib

controls = importlib.import_module("scripts.analyze_av_event_input_controls")


def _row(row_id, video_id, sound_id):
    return {"id": row_id, "video_id": video_id, "sound_id": sound_id}


def test_cross_video_audio_donors_are_deterministic_and_never_self_video():
    rows = [
        _row("a0", "video_a", 0),
        _row("a1", "video_a", 1),
        _row("b0", "video_b", 0),
        _row("c0", "video_c", 0),
    ]
    first = controls.cross_video_audio_donors(rows, seed=17)
    second = controls.cross_video_audio_donors(rows, seed=17)
    assert first == second
    assert len(first) == len(rows)
    donor_by_id = {row["id"]: row["video_id"] for row in rows}
    assert all(donor_by_id[first[row["id"]]] != row["video_id"] for row in rows)


def test_stale_audio_uses_only_a_previous_event_from_the_same_video():
    rows = [
        _row("a0", "video_a", 0),
        _row("a1", "video_a", 1),
        _row("a2", "video_a", 2),
        _row("b0", "video_b", 0),
    ]
    assert controls.previous_event_audio_donors(rows) == {"a1": "a0", "a2": "a1"}


def test_stale_audio_order_uses_actual_sample_start_when_available():
    rows = [
        {**_row("later", "video_a", 0), "audio_sample_start": 200},
        {**_row("earlier", "video_a", 9), "audio_sample_start": 100},
    ]
    assert controls.previous_event_audio_donors(rows) == {"later": "earlier"}
