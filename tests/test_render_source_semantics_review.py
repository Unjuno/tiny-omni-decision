import pytest

pytest.importorskip("av")
pytest.importorskip("torch")
from scripts.render_source_semantics_review import sample_times  # noqa: E402


def test_sample_times_within_event_and_invalid_interval():
    assert sample_times([1_000_000, 2_000_000]) == [1.1, 1.5, 1.9]
    with pytest.raises(ValueError):
        sample_times([100, 100])
