import pytest

pytest.importorskip("av")
pytest.importorskip("torch")


def test_exclusive_cut_and_no_cut_sentinel():
    from scripts.extract_occlusion_videomaev2_features import check_cut

    check_cut([0, 20], -1)
    check_cut([0, 20], 21)
    with pytest.raises(ValueError, match="cut"):
        check_cut([0, 21], 21)
    with pytest.raises(ValueError, match="cut"):
        check_cut([0, 1], -2)
