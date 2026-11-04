import pytest

pytest.importorskip("numpy")
from scripts.extract_starss23_features import validate_windows


def test_feature_extraction_preserves_order_and_identity():
    rows = [{"id": "scene:a", "split": "dev-train"}, {"id": "scene:b", "split": "dev-train"}]
    assert validate_windows({"status": "PASS", "windows": rows}) is rows


@pytest.mark.parametrize(
    "rows", [[], [{"id": "a", "split": "validation"}], [{"id": "a", "split": "dev-train"}] * 2]
)
def test_feature_extraction_rejects_invalid_or_evaluation_windows(rows):
    with pytest.raises(ValueError):
        validate_windows({"status": "PASS", "windows": rows})
