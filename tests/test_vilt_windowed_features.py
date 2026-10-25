import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")


def _tiny_processor(tmp_path):
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocabulary.extend(f"token{index}" for index in range(64))
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary), encoding="utf-8")
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
    image_processor = transformers.ViltImageProcessor(size=32, size_divisor=16)
    return transformers.ViltProcessor(image_processor=image_processor, tokenizer=tokenizer)


def test_windowed_feature_encoding_matches_direct_short_forward_and_means_long_windows(tmp_path):
    from scripts.extract_clevr_vqa_features import encode_windowed_images
    from tiny_omni_decision.vilt_compat import (
        build_vilt_question_input_ids,
        split_vilt_token_ids,
    )

    torch.manual_seed(17)
    processor = _tiny_processor(tmp_path)
    config = transformers.ViltConfig(
        vocab_size=len(processor.tokenizer),
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        image_size=32,
        patch_size=16,
        max_image_length=4,
        max_position_embeddings=40,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model = transformers.ViltForQuestionAnswering(config).eval()
    image = Image.new("RGB", (32, 32), color=(200, 20, 10))
    try:
        short_question = "token1 token2 token3"
        short_ids = build_vilt_question_input_ids(processor.tokenizer, short_question)
        short_windows = split_vilt_token_ids(
            short_ids,
            max_positions=40,
            overlap_tokens=8,
            cls_token_id=processor.tokenizer.cls_token_id,
            sep_token_id=processor.tokenizer.sep_token_id,
        )
        actual_short = encode_windowed_images(
            model, processor, [image], [short_windows], "cpu"
        )
        direct_inputs = processor(
            images=[image],
            text=[short_question],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        with torch.inference_mode():
            direct_short = model.vilt(**direct_inputs).pooler_output
        assert torch.allclose(actual_short, direct_short, atol=1e-6, rtol=1e-6)

        long_question = " ".join(f"token{index}" for index in range(45))
        long_ids = build_vilt_question_input_ids(processor.tokenizer, long_question)
        long_windows = split_vilt_token_ids(
            long_ids,
            max_positions=40,
            overlap_tokens=8,
            cls_token_id=processor.tokenizer.cls_token_id,
            sep_token_id=processor.tokenizer.sep_token_id,
        )
        assert [len(window) for window in long_windows] == [40, 17]
        actual_long = encode_windowed_images(
            model, processor, [image], [long_windows], "cpu"
        )
        independent_windows = torch.cat(
            [
                encode_windowed_images(model, processor, [image], [[window]], "cpu")
                for window in long_windows
            ],
            dim=0,
        )
        assert actual_long.shape == (1, 24)
        assert torch.isfinite(actual_long).all()
        assert torch.allclose(
            actual_long, independent_windows.mean(dim=0, keepdim=True), atol=1e-6, rtol=1e-6
        )
    finally:
        image.close()
