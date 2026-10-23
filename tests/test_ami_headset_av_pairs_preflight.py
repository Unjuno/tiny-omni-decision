from __future__ import annotations

from scripts.preflight_ami_headset_av_pairs import (
    MEETINGS,
    make_session,
    select_one_per_speaker,
)


def _row(
    meeting: str, speaker: str, index: int, begin: float, duration: float
) -> dict[str, object]:
    return {
        "meeting_id": meeting,
        "speaker_id": speaker,
        "microphone_id": "H00",
        "audio_id": f"{meeting}_{speaker}_{index}",
        "source_row_idx": index,
        "begin_time": begin,
        "end_time": begin + duration,
        "duration": duration,
    }


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    rows_by_meeting: dict[str, list[dict[str, object]]] = {}
    videos: list[dict[str, object]] = []
    for meetings in MEETINGS.values():
        for meeting in meetings:
            rows_by_meeting[meeting] = [
                _row(meeting, f"speaker-{i}", i, 10 + i * 20, 2.0 + i / 10)
                for i in range(4)
            ]
            videos.extend(
                {"filename": f"{meeting}.Closeup{i}.rm", "duration_s": 100.0}
                for i in range(1, 5)
            )
    return {"rows_by_meeting": rows_by_meeting}, {"results": videos}


def test_selects_exactly_one_deterministic_row_per_speaker_and_meeting() -> None:
    metadata, video_receipt = _fixture()
    first = select_one_per_speaker(metadata, video_receipt)
    second = select_one_per_speaker(metadata, video_receipt)

    assert len(first) == 24
    assert [row["audio_id"] for row in first] == [row["audio_id"] for row in second]
    for meeting_ids in MEETINGS.values():
        for meeting in meeting_ids:
            rows = [row for row in first if row["meeting_id"] == meeting]
            assert len(rows) == 4
            assert len({row["speaker_id"] for row in rows}) == 4


def test_selection_respects_audio_duration_and_video_tail_margins() -> None:
    metadata, video_receipt = _fixture()
    rows = metadata["rows_by_meeting"]["ES2008b"]
    rows.append(_row("ES2008b", "speaker-0", 100, 2080.0, 4.0))
    rows.append(_row("ES2008b", "speaker-1", 101, 5.0, 1.2))

    selected = select_one_per_speaker(metadata, video_receipt)
    es_rows = [row for row in selected if row["meeting_id"] == "ES2008b"]

    assert len(es_rows) == 4
    assert all(float(row["end_time"]) <= 2083.0 for row in es_rows)
    assert all(float(row["duration"]) >= 1.5 for row in es_rows)
    assert all(int(row["source_row_idx"]) < 100 for row in es_rows)


def test_rejects_missing_meeting_camera_coverage() -> None:
    metadata, video_receipt = _fixture()
    video_receipt["results"] = [
        row for row in video_receipt["results"] if row["filename"] != "TS3009a.Closeup1.rm"
    ]

    try:
        select_one_per_speaker(metadata, video_receipt)
    except ValueError as exc:
        assert "expected four verified video durations" in str(exc)
    else:
        raise AssertionError("missing camera coverage must fail closed")


def test_http_retry_is_bounded_to_idempotent_gets_and_transient_statuses() -> None:
    session = make_session()
    retry = session.get_adapter("https://").max_retries

    assert retry.total == 3
    assert retry.connect == 3
    assert retry.read == 2
    assert retry.status_forcelist == (429, 500, 502, 503, 504)
    assert retry.allowed_methods == frozenset({"GET"})
    session.close()
