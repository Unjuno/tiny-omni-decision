import pytest

from scripts.train_text_coverage import check_order


def test_fixed_order_must_cover_unique_existing_ids():
    assert check_order(["a", "b"], {"a": {}, "b": {}}, 2) is None
    with pytest.raises(ValueError):
        check_order(["a", "a"], {"a": {}}, 2)
    with pytest.raises(ValueError):
        check_order(["a", "c"], {"a": {}, "b": {}}, 2)
    with pytest.raises(ValueError):
        check_order(["a"], {"a": {}}, 2)
