import pytest

torch = pytest.importorskip("torch")


def test_options_are_deterministic_unique_and_include_target():
    from tiny_omni_decision.clevr_vqa_options import ANSWER_VOCABULARIES, make_options

    for _kind, vocabulary in ANSWER_VOCABULARIES.items():
        for answer in vocabulary:
            first = make_options(
                image_filename="CLEVR_train_000017.png",
                question_index=9,
                answer=answer,
            )
            assert first == make_options(
                image_filename="CLEVR_train_000017.png",
                question_index=9,
                answer=answer,
            )
            assert answer in first
            assert len(first) == len(set(first))
            assert 2 <= len(first) <= 4
            assert set(first) <= set(vocabulary)


def test_options_change_with_example_identity():
    from tiny_omni_decision.clevr_vqa_options import make_options

    base = make_options(
        image_filename="CLEVR_train_000017.png", question_index=9, answer="5"
    )
    other_image = make_options(
        image_filename="CLEVR_train_000018.png", question_index=9, answer="5"
    )
    assert base != other_image


def test_option_probabilities_mask_to_supplied_labels_and_keep_order():
    from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS, option_probabilities

    logits = torch.zeros(len(ANSWER_LABELS))
    logits[ANSWER_LABELS.index("cyan")] = 3.0
    logits[ANSWER_LABELS.index("red")] = 1.0
    probabilities = option_probabilities(logits, ["red", "cyan"])
    reversed_probabilities = option_probabilities(logits, ["cyan", "red"])

    torch.testing.assert_close(probabilities, reversed_probabilities.flip(0))
    assert probabilities[1] > probabilities[0]
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_option_probabilities_reject_invalid_labels_shapes_and_duplicates():
    from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS, option_probabilities

    logits = torch.zeros(len(ANSWER_LABELS))
    with pytest.raises(ValueError, match="one logit"):
        option_probabilities(logits[:-1], ["yes", "no"])
    with pytest.raises(ValueError, match="unique"):
        option_probabilities(logits, ["yes", "yes"])
    with pytest.raises(ValueError, match="outside"):
        option_probabilities(logits, ["violet", "purple"])
    with pytest.raises(ValueError, match="between two and four"):
        option_probabilities(logits, ["yes"])
    with pytest.raises(ValueError, match="between two and four"):
        option_probabilities(logits, ["no", "yes", "cube", "small", "red"])
    with pytest.raises(ValueError, match="share one answer kind"):
        option_probabilities(logits, ["yes", "cube"])
    logits[0] = float("nan")
    with pytest.raises(FloatingPointError, match="finite"):
        option_probabilities(logits, ["yes", "no"])
