import pytest

torch = pytest.importorskip("torch")


def _api():
    from tiny_omni_decision.textfree_video import (
        TextFreeVideoReadout,
        build_video_vocab,
        encode_video_rows,
    )

    return TextFreeVideoReadout, build_video_vocab, encode_video_rows


def _rows():
    return [
        {
            "id": "scene-a-q1",
            "media_sha256": "a" * 64,
            "scene_group_id": "scene-a",
            "question": "How many red objects collide?",
            "program": ["events", "filter_color", "red", "count"],
            "taxonomy": "count",
            "question_type": "temporal_descriptive",
            "options": ["0", "1", "2"],
            "target": "1",
        }
    ]


@pytest.mark.parametrize("temporal_mode", ["mean_mlp", "bilstm"])
def test_textfree_video_model_scores_only_supplied_options_and_backpropagates(temporal_mode):
    model_type, build_vocab, encode_rows = _api()
    vocab = build_vocab(_rows())
    encoded = encode_rows(_rows(), {"a" * 64: torch.randn(8, 768)}, vocab)
    model = model_type(vocab, temporal_mode=temporal_mode)

    logits = model(
        encoded.video,
        encoded.program_ids,
        encoded.taxonomy_ids,
        encoded.option_ids,
        encoded.option_mask,
    )
    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, encoded.targets).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_video_input_builder_ignores_natural_language_question_text():
    _, build_vocab, encode_rows = _api()
    rows = _rows()
    vocab = build_vocab(rows)
    features = {"a" * 64: torch.zeros(8, 768)}
    original = encode_rows(rows, features, vocab)
    changed = _rows()
    changed[0]["question"] = "leaked candidate answer must not be encoded"
    altered = encode_rows(changed, features, vocab)

    assert torch.equal(original.program_ids, altered.program_ids)
    assert torch.equal(original.taxonomy_ids, altered.taxonomy_ids)
    assert torch.equal(original.option_ids, altered.option_ids)
    assert torch.equal(original.targets, altered.targets)


def test_validation_only_program_and_option_tokens_map_to_unknown():
    _, build_vocab, encode_rows = _api()
    vocab = build_vocab(_rows())
    row = _rows()[0]
    row["program"] = ["unseen_validation_operation"]
    row["options"] = ["unseen-a", "unseen-b"]
    row["target"] = "unseen-a"

    encoded = encode_rows([row], {"a" * 64: torch.zeros(8, 768)}, vocab)

    assert (encoded.program_ids == vocab.program_to_id["<unk>"]).all()
    assert (encoded.option_ids == vocab.option_to_id["<unk>"]).all()
    assert encoded.targets.item() == 0


def test_textfree_video_vocab_is_deterministic_and_uses_training_rows_only():
    _, build_vocab, _ = _api()
    training = _rows()
    first = build_vocab(training)
    second = build_vocab(list(reversed(training)))
    assert first == second
    assert "leaked" not in first.program_to_id


def test_parameter_matched_temporal_blocks_stay_within_one_percent():
    model_type, build_vocab, _ = _api()
    vocab = build_vocab(_rows())
    mean = model_type(vocab, temporal_mode="mean_mlp")
    recurrent = model_type(vocab, temporal_mode="bilstm")
    mean_count = sum(parameter.numel() for parameter in mean.parameters())
    recurrent_count = sum(parameter.numel() for parameter in recurrent.parameters())
    assert abs(mean_count - recurrent_count) / max(mean_count, recurrent_count) < 0.01


def test_candidate_order_permutation_preserves_option_scores():
    model_type, build_vocab, encode_rows = _api()
    vocab = build_vocab(_rows())
    row = _rows()[0]
    features = {"a" * 64: torch.arange(8 * 768, dtype=torch.float32).reshape(8, 768)}
    torch.manual_seed(17)
    model = model_type(vocab, temporal_mode="bilstm").eval()
    first = encode_rows([row], features, vocab)
    permuted_row = {**row, "options": ["2", "0", "1"]}
    permuted = encode_rows([permuted_row], features, vocab)
    with torch.inference_mode():
        first_logits = model(
            first.video,
            first.program_ids,
            first.taxonomy_ids,
            first.option_ids,
            first.option_mask,
        )
        permuted_logits = model(
            permuted.video,
            permuted.program_ids,
            permuted.taxonomy_ids,
            permuted.option_ids,
            permuted.option_mask,
        )
    assert torch.allclose(first_logits[:, [2, 0, 1]], permuted_logits, atol=1e-6, rtol=0)


def test_video_split_gate_rejects_scene_media_and_normalized_content_overlap():
    from tiny_omni_decision.textfree_video import validate_video_splits

    train = _rows()
    validation = _rows()
    validation[0]["id"] = "other-id"
    validation[0]["scene_group_id"] = "other-scene"
    validation[0]["media_sha256"] = "b" * 64
    validation[0]["question"] = "  HOW   MANY RED OBJECTS COLLIDE? "

    with pytest.raises(ValueError, match="normalized question/options overlap"):
        validate_video_splits(train, validation)

    validation[0]["question"] = "How many blue objects collide?"
    validation[0]["media_sha256"] = train[0]["media_sha256"]
    with pytest.raises(ValueError, match="scene group overlap|media identity overlap"):
        validate_video_splits(train, validation)
