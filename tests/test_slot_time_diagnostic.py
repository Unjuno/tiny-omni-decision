import pytest

torch = pytest.importorskip("torch")
from scripts.diagnose_slot_time import perturb  # noqa: E402


@pytest.mark.parametrize("mode", ["normal", "reverse_time", "zero_time"])
def test_only_timestamp_changes_without_mutating_source(mode):
    value = torch.arange(224 * 129, dtype=torch.float32).reshape(224, 129)
    value[:, -1] = torch.linspace(0, 1, 224)
    original = value.clone()
    changed = perturb({"video": value}, mode)["video"]
    assert torch.equal(value, original)
    assert changed.data_ptr() != value.data_ptr()
    assert torch.equal(changed[:, :-1], original[:, :-1])
    expected = (1 - original[:, -1] if mode == "reverse_time" else
                torch.zeros(224) if mode == "zero_time" else original[:, -1])
    assert torch.equal(changed[:, -1], expected)


def test_unknown_perturbation_rejected():
    with pytest.raises(ValueError, match="unknown"):
        perturb({}, "shuffle")
