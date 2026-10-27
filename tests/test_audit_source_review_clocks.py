import pytest

pytest.importorskip("av")
pytest.importorskip("torch")
from scripts.audit_source_review_clocks import first_video_frame  # noqa: E402


def test_invalid_interval_rejected_before_opening_file(tmp_path):
    with pytest.raises(ValueError, match="invalid event"):
        first_video_frame(tmp_path / "missing.mp4", 10, 10)
    with pytest.raises(ValueError, match="invalid event"):
        first_video_frame(tmp_path / "missing.mp4", -1, 10)
