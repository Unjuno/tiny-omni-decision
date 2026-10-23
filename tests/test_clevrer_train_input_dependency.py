import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.audit_clevrer_train_input_dependency import (  # noqa: E402
    apply_control,
    pair_donor_rows,
)


def fixture():
    return {
        "pair_ids": np.array(["b", "a", "c", "a", "c", "b"]),
        "scene_ids": np.array(["s2", "s1", "s3", "s1", "s3", "s2"]),
        "targets": np.array([0, 1, 1, 0, 0, 1]),
    }


def test_pair_donors_are_deterministic_bijective_and_cross_scene():
    metadata = fixture()
    rows = pair_donor_rows(metadata)
    assert rows.shape == (6,)
    assert sorted(rows.tolist()) == list(range(6))
    assert not np.any(metadata["scene_ids"] == metadata["scene_ids"][rows])
    assert np.array_equal(rows, pair_donor_rows(metadata))
    for pair in ["a", "b", "c"]:
        members = np.flatnonzero(metadata["pair_ids"] == pair)
        assert {metadata["targets"][i] for i in members} == {0, 1}
        for i in members:
            assert metadata["targets"][i] == metadata["targets"][rows[i]]
            assert metadata["scene_ids"][i] != metadata["scene_ids"][rows[i]]
            assert metadata["pair_ids"][i] != metadata["pair_ids"][rows[i]]


def test_rejects_incomplete_or_mixed_scene_pairs_and_single_scene_pool():
    metadata = fixture()
    metadata["scene_ids"][0] = "s1"
    with pytest.raises(ValueError, match="same scene"):
        pair_donor_rows(metadata)
    metadata = fixture()
    metadata["targets"][0] = 1
    with pytest.raises(ValueError, match="both target"):
        pair_donor_rows(metadata)
    metadata = fixture()
    metadata["scene_ids"][:] = "same"
    with pytest.raises(ValueError, match="different-scene"):
        pair_donor_rows(metadata)


def test_input_controls_are_immutable_and_use_only_the_named_stream():
    values = {k: torch.arange(12).reshape(3, 2, 2).float()
              for k in ["audio", "video"]}
    values["text"] = torch.arange(6).reshape(3, 2).float()
    donors = torch.tensor([2, 0, 1])
    originals = {k: v.clone() for k, v in values.items()}
    for name, key in [("zero_audio", "audio"), ("zero_video", "video"),
                       ("shuffle_audio", "audio"), ("shuffle_video", "video"),
                       ("reverse_video", "video")]:
        changed = apply_control(values, name, donors)
        if name.startswith("zero"):
            torch.testing.assert_close(changed[key], torch.zeros_like(values[key]))
        elif name.startswith("shuffle"):
            torch.testing.assert_close(changed[key], values[key][donors])
        else:
            torch.testing.assert_close(changed[key], values[key].flip(1))
        for other in values:
            torch.testing.assert_close(values[other], originals[other])
            if other != key:
                assert changed[other] is values[other]
    with pytest.raises(ValueError, match="unknown"):
        apply_control(values, "drop_text", donors)
