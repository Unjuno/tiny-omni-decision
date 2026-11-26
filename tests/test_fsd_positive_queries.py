import pytest

from scripts.fsd_positive_queries import positive_queries


def row():
    return {"id": "123", "uploader": "author", "event": "Bell",
            "license": "http://creativecommons.org/publicdomain/zero/1.0/",
            "known_positive_labels": "Bell,Music"}


def test_unknown_events_never_become_negative_queries():
    queries = positive_queries([row()])
    assert len(queries) == 1
    assert queries[0]["target"] == 1
    assert queries[0]["scope"] == "whole_clip"
    assert queries[0]["unknown_events"] == "omitted"


@pytest.mark.parametrize("change", [
    {"known_positive_labels": "Music"}, {"license": "unknown"}, {"event": "Music"},
    {"id": "../123"}, {"uploader": ""},
])
def test_invalid_positive_or_rights_fails_closed(change):
    with pytest.raises(ValueError):
        positive_queries([{**row(), **change}])


def test_duplicate_clip_or_uploader_rejected():
    for change in ({}, {"id": "456"}, {"uploader": "another"}):
        with pytest.raises(ValueError):
            positive_queries([row(), {**row(), **change}])
