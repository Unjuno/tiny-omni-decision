import pytest

from scripts.acquire_av_correspondence import validate_selection


def test_acquisition_preserves_large_preselected_member_and_rejects_bad_split():
    row = dict(video_id="v1", bytes=122791611, metadata={"split": "valid"})
    validate_selection([row], 1, "valid")
    with pytest.raises(ValueError, match="split"):
        validate_selection([row], 1, "train")
    with pytest.raises(ValueError, match="duplicate"):
        validate_selection([row, row], 2, "valid")
    with pytest.raises(ValueError, match="cap"):
        validate_selection([dict(row, bytes=256000001)], 1, "valid")
