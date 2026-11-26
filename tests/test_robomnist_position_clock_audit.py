from scripts.fetch_robomnist_position_clock_audit import extract_timestamp_summaries


def test_timestamp_summary_omits_all_position_values():
    value = {
        "samples": [
            {"time_stamp_ns": 0, "position": [1.25, -4.0, 9.0]},
            {"time_stamp_ns": 33_000_000, "position": [8.0, 2.0, 0.5]},
        ],
        "metadata": {"capture_time": "2026-01-01T00:00:00Z"},
    }

    result = extract_timestamp_summaries(value)

    assert result == [
        {
            "path": "samples[].time_stamp_ns",
            "kind": "numeric",
            "count": 2,
            "minimum": 0.0,
            "maximum": 33_000_000.0,
            "strictly_increasing": True,
        },
        {
            "path": "metadata.capture_time",
            "kind": "text",
            "count": 1,
            "minimum": None,
            "maximum": None,
            "strictly_increasing": None,
        },
    ]
    serialized = repr(result)
    assert "position" not in serialized
    assert "1.25" not in serialized


def test_timestamp_summary_ignores_non_timestamp_numbers():
    assert extract_timestamp_summaries({"position": [1, 2, 3], "label": 7}) == []


def test_timestamp_summary_flags_nonmonotonic_clock_values():
    result = extract_timestamp_summaries({"timestamps": [0, 2, 1]})
    assert result[0]["strictly_increasing"] is False
