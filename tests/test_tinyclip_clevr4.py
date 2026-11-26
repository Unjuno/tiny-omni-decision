import pytest

from scripts.evaluate_tinyclip_clevr4 import prompts, rotated_indices


def test_fixed_prompts_and_image_rotation():
    assert prompts("count", ["1", "10"]) == [
        "An image containing 1 objects.",
        "An image containing 10 objects.",
    ]
    assert prompts("shape", ["cube"]) == ["An image of cube-shaped objects."]
    assert rotated_indices(["a", "b", "c"]) == [1, 2, 0]
    ids = [str(i) for i in range(512)]
    rotation = rotated_indices(ids)
    assert sorted(rotation) == list(range(512))
    assert all(i != j for i, j in enumerate(rotation))
    for bad in ([], ["a"], ["a", "a"]):
        with pytest.raises(ValueError):
            rotated_indices(bad)
