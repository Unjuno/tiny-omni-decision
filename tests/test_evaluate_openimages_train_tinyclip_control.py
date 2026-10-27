import pytest

pytest.importorskip("torch")
from scripts.evaluate_openimages_train_tinyclip_control import fixed_derangement  # noqa: E402


def test_derangement_is_fixed_and_has_no_self_donors():
    values = fixed_derangement(257)
    assert values == fixed_derangement(257)
    assert sorted(values) == list(range(257))
    assert all(index != donor for index, donor in enumerate(values))
