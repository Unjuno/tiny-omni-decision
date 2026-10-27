import pytest

pytest.importorskip("torch")
from scripts.freeze_source_semantics_review import STRATA, choose  # noqa: E402


def test_sample_is_order_independent_and_video_disjoint():
    rows = [
        {"category": s, "video_id": f"{s}-{i}", "record_id": f"{s}-{i}:a"}
        for s in STRATA
        for i in range(6)
    ]
    result = choose(rows)
    assert result == choose(reversed(rows))
    assert len(result) == 16 and len({r["video_id"] for r in result}) == 16


def test_missing_stratum_fails_closed():
    with pytest.raises(ValueError, match="distinct videos"):
        choose([])
