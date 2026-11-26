import json

import pytest


def test_json_report_comparison_handles_integer_count_keys_without_rounding():
    from scripts.lombard_correspondence_reports import same_report

    value = {"distances": {1: 2, 3: 8}, "metric": 0.1234567890123456}
    saved = json.loads(json.dumps(value))
    assert value != saved
    assert same_report(value, saved)
    saved["metric"] += 1e-15
    assert not same_report(value, saved)
    with pytest.raises(ValueError):
        same_report({"metric": float("nan")}, {})
