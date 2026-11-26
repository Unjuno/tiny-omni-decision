import numpy as np
import pytest

from tiny_omni_decision import fine_qa_inputs as module


def test_scene_split_is_deterministic_and_media_content_cannot_leak():
    ids = [f"video_{i}" for i in range(99)]
    hashes = [f"hash{i}" for i in range(99)]
    train, hold = module.scene_split(ids, hashes)
    assert len(train) == 79 and len(hold) == 20
    assert not set(train) & set(hold) and set(train) | set(hold) == set(range(99))
    assert all(
        np.array_equal(a, b)
        for a, b in zip((train, hold), module.scene_split(ids, hashes), strict=True)
    )
    reverse = list(range(98, -1, -1))
    _, other = module.scene_split([ids[i] for i in reverse], [hashes[i] for i in reverse])
    assert {ids[i] for i in hold} == {ids[reverse[i]] for i in other}
    hashes[-1] = hashes[0]
    with pytest.raises(ValueError):
        module.scene_split(ids, hashes)


def test_question_subset_keeps_both_conditions_three_options_and_original_order():
    pairs = [
        dict(
            video_id=f"video_{i}",
            split="train",
            options=["a", "b", "c"],
            questions=[
                dict(condition="initial", target=i % 3),
                dict(condition="final", target=(i + 1) % 3),
            ],
        )
        for i in range(3)
    ]
    text = np.arange(6 * 3 * 1536, dtype=np.float32).reshape(6, 3, 1536)
    result = module.paired_subset(pairs, text, np.array([2, 0], np.int64))
    assert [r["video_id"] for r in result["pairs"]] == ["video_2", "video_0"]
    np.testing.assert_array_equal(result["text"], text[[4, 5, 0, 1]])
    assert result["targets"].tolist() == [2, 0, 0, 1]
    with pytest.raises(ValueError):
        module.paired_subset(pairs, text, np.array([0, 0], np.int64))


def test_context_controls_keep_native_coordinates_and_do_not_mutate_inputs():
    appearance = np.arange(2 * 4 * 384, dtype=np.float32).reshape(2, 4, 384)
    positions = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    before = appearance.tobytes()
    normal = module.context_view(appearance, positions)
    assert normal.shape == (8, 387)
    for control, value in [
        ("initial", np.broadcast_to(appearance[:1], appearance.shape)),
        ("reverse", appearance[::-1]),
    ]:
        result = module.context_view(appearance, positions, control)
        np.testing.assert_array_equal(result[:, :384], value.reshape(8, 384))
        np.testing.assert_array_equal(result[:, 384:], positions.reshape(8, 3))
    assert not module.context_view(appearance, positions, "zero").any()
    assert appearance.tobytes() == before


@pytest.mark.parametrize("bad", ["id", "count", "seed"])
def test_invalid_scene_contract_fails_closed(bad):
    ids = ["a", "b", "c"]
    hashes = ["x", "y", "z"]
    held = 1
    seed = 17
    if bad == "id":
        ids[-1] = ids[0]
    elif bad == "count":
        held = 3
    else:
        seed = True
    with pytest.raises(ValueError):
        module.scene_split(ids, hashes, held_out=held, seed=seed)


def test_invalid_context_alignment_or_unknown_control_rejected():
    a = np.zeros((2, 4, 384), np.float32)
    p = np.zeros((2, 4, 3), np.float32)
    for value, position, control in [
        (a, p[:1], "normal"),
        (a, p, "unrecognized"),
        (a.astype(float), p, "normal"),
    ]:
        with pytest.raises(ValueError):
            module.context_view(value, position, control)
