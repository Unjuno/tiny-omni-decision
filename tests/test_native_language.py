import pytest

torch = pytest.importorskip("torch")


def module():
    from tiny_omni_decision import native_language
    return native_language


@pytest.mark.parametrize("count", [0, 1, 510, 511, 753, 824, 8190])
def test_every_content_token_retained_once(count):
    spans = module().content_spans(count)
    kept = [i for start, stop, keep in spans for i in range(start + keep, stop)]
    assert kept == list(range(count))
    assert all(stop - start + 2 <= 512 for start, stop, _ in spans)
    assert all(spans[i][0] == spans[i - 1][1] - 32 for i in range(1, len(spans)))


@pytest.mark.parametrize("count", [-1, 8191, True, 2.5])
def test_invalid_or_over_budget_length_fails_closed(count):
    with pytest.raises(ValueError):
        module().content_spans(count)


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, text_pair=None, **kwargs):
        ids = [int(x) for x in text.split()]
        if text_pair is not None:
            ids = ids + [2] + [int(x) for x in text_pair.split()]
        ids = [1, *ids, 2]
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids),
                    dtype=torch.long), token_type_ids=torch.zeros(1, len(ids), dtype=torch.long))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls = []

    def deberta(self, **encoded):
        self.calls.append(encoded['input_ids'].shape[1])
        assert self.calls[-1] <= 512
        values = encoded['input_ids'].float().unsqueeze(-1).repeat(1, 1, 384)
        return type('Output', (), {'last_hidden_state': values})()


def test_short_state_uses_one_original_forward():
    model = Model().eval()
    hidden, spans = module().observation_tokens(model, Tokenizer(), '3 4 5')
    assert model.calls == [5]
    assert spans == [(0, 3, 0)]
    assert hidden[:, 0].tolist() == [1, 3, 4, 5, 2]


def test_long_state_contains_all_tokens_without_overlap_or_special_duplicates():
    model = Model().eval()
    ids = list(range(3, 827))
    hidden, spans = module().observation_tokens(model, Tokenizer(), ' '.join(map(str, ids)))
    assert model.calls == [512, 348]
    assert len(spans) == 2
    assert hidden.shape == (826, 384)
    assert hidden[:, 0].tolist() == [1, *ids, 2]


def test_question_candidate_mean_includes_original_special_tokens():
    value = module().candidate_vector(Model().eval(), Tokenizer(), '3 4', '5 6')
    assert value.shape == (384,)
    torch.testing.assert_close(value, torch.full((384,), 23 / 7))


def test_query_pair_cannot_be_silently_windowed():
    with pytest.raises(ValueError):
        module().candidate_vector(Model().eval(), Tokenizer(), '3 ' * 510, '4')


def test_encoder_must_be_eval_and_frozen():
    model = Model()
    with pytest.raises(ValueError):
        module().observation_tokens(model, Tokenizer(), '3')
    model.eval().parameter.requires_grad_(True)
    with pytest.raises(ValueError):
        module().observation_tokens(model, Tokenizer(), '3')


@pytest.mark.parametrize('field,value', [('attention_mask', 0), ('token_type_ids', 1),
                                       ('input_ids', 99)])
def test_observation_rejects_unrecognized_single_sequence_layout(field, value):
    class InvalidTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result[field][0, 0] = value
            return result

    with pytest.raises(ValueError):
        module().observation_tokens(Model().eval(), InvalidTokenizer(), '3 4')


def test_nonfinite_model_output_is_not_cached():
    class NonfiniteModel(Model):
        def deberta(self, **encoded):
            result = super().deberta(**encoded)
            result.last_hidden_state[0, 0, 0] = float('nan')
            return result

    with pytest.raises(ValueError):
        module().candidate_vector(NonfiniteModel().eval(), Tokenizer(), '3', '4')
