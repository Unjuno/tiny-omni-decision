import pytest

torch = pytest.importorskip("torch")


def test_only_expected_deterministic_buffer_is_removed():
    from tiny_omni_decision.vilt_compat import remove_legacy_position_ids

    key = "vilt.embeddings.text_embeddings.position_ids"
    weight = torch.randn(2, 3)
    original = {key: torch.arange(40).unsqueeze(0), "weight": weight}
    translated = remove_legacy_position_ids(original, 40)
    assert translated == {"weight": weight}
    assert key in original
    for values in [
        torch.arange(39).unsqueeze(0),
        torch.zeros(1, 40, dtype=torch.int64),
        torch.arange(40, dtype=torch.float32).unsqueeze(0),
    ]:
        with pytest.raises(ValueError):
            remove_legacy_position_ids({key: values}, 40)


def test_long_questions_are_losslessly_split_with_overlapping_special_tokens():
    from tiny_omni_decision.vilt_compat import split_vilt_token_ids

    input_ids = [101, *range(200, 247), 102]
    windows = split_vilt_token_ids(
        input_ids,
        max_positions=40,
        overlap_tokens=8,
        cls_token_id=101,
        sep_token_id=102,
    )

    assert [len(window) for window in windows] == [40, 19]
    assert all(window[0] == 101 and window[-1] == 102 for window in windows)
    assert windows[0][-9:-1] == windows[1][1:9]
    assert windows[0][1:-1] + windows[1][9:-1] == input_ids[1:-1]


def test_short_questions_keep_the_original_single_window():
    from tiny_omni_decision.vilt_compat import split_vilt_token_ids

    input_ids = [101, 12, 34, 102]
    assert split_vilt_token_ids(
        input_ids,
        max_positions=40,
        overlap_tokens=8,
        cls_token_id=101,
        sep_token_id=102,
    ) == [input_ids]


def test_question_tokenization_preserves_the_full_text_and_vilt_boundaries(tmp_path):
    transformers = pytest.importorskip("transformers")
    from tiny_omni_decision.vilt_compat import build_vilt_question_input_ids

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab.extend(f"token{index}" for index in range(50))
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=False,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=40,
    )
    question = " ".join(f"token{index}" for index in range(45))

    input_ids = build_vilt_question_input_ids(tokenizer, question)

    assert len(input_ids) == 47
    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[-1] == tokenizer.sep_token_id
    assert input_ids[1:-1] == [tokenizer.convert_tokens_to_ids(f"token{i}") for i in range(45)]
    assert tokenizer.model_max_length == 40


@pytest.mark.parametrize(
    "max_positions,overlap_tokens",
    [(2, 0), (40, -1), (40, 38), (40, 40)],
)
def test_windowing_rejects_invalid_capacity_or_overlap(max_positions, overlap_tokens):
    from tiny_omni_decision.vilt_compat import split_vilt_token_ids

    with pytest.raises(ValueError):
        split_vilt_token_ids(
            [101, 12, 34, 102],
            max_positions=max_positions,
            overlap_tokens=overlap_tokens,
            cls_token_id=101,
            sep_token_id=102,
        )
