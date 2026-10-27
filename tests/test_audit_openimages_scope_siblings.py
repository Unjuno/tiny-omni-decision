import pytest

pytest.importorskip("torch")
from scripts.audit_openimages_scope_siblings import inspect  # noqa: E402


def test_only_other_frozen_siblings_are_counted():
    result = inspect({"a", "b", "c"}, {"a": ("a", "b", "d"), "b": ("c",)})
    assert result == {"a": 1, "b": 1, "c": 0}
    with pytest.raises(ValueError):
        inspect(set(), {})
