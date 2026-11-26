import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from test_sam2_batched_decoder_contract import fixture  # noqa: E402

from tiny_omni_decision import sam2_chunked_masks as module  # noqa: E402
from tiny_omni_decision.sam2_batch_masks import initial_mask_decode  # noqa: E402


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 4, 9])
def test_all_ordered_objects_and_tail_match_actual_native_heads(chunk_size):
    model, images, points, labels = fixture()
    with torch.inference_mode():
        expected = initial_mask_decode(model, images, points, labels)
        actual = module.chunked_initial_mask_decode(model, images, points, labels, chunk_size)
    assert actual[0].shape == (4, 16, 16) and actual[1].shape == (4,)
    assert (actual[0] != -1024.).any()
    assert torch.allclose(actual[0], expected[0], atol=1e-4, rtol=0)
    assert torch.allclose(actual[1], expected[1], atol=1e-5, rtol=0)
    assert torch.equal(actual[0] > 0, expected[0] > 0)


@pytest.mark.parametrize('chunk_size', [0, -1, True, 2.5, '2', None])
def test_invalid_chunk_size_fails_before_any_head_call(chunk_size):
    model, images, points, labels = fixture()
    calls = []
    hook = model.prompt_encoder.register_forward_pre_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError):
        module.chunked_initial_mask_decode(model, images, points, labels, chunk_size)
    hook.remove()
    assert calls == []


@pytest.mark.parametrize('kind', ['empty', 'bad_final_label', 'final_nan', 'misaligned',
                                 'multi_image', 'float_labels', 'half_points'])
def test_entire_prompt_contract_admitted_before_partial_inference(kind):
    model, images, points, labels = fixture()
    if kind == 'empty':
        points, labels = points[:, :0], labels[:, :0]
    elif kind == 'bad_final_label':
        labels[:, -1] = 0
    elif kind == 'final_nan':
        points[:, -1] = torch.nan
    elif kind == 'misaligned':
        labels = labels[:, :-1]
    elif kind == 'multi_image':
        points, labels = points.repeat(2, 1, 1, 1), labels.repeat(2, 1, 1)
    elif kind == 'float_labels':
        labels = labels.float()
    else:
        points = points.half()
    calls = []
    hook = model.prompt_encoder.register_forward_pre_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError):
        module.chunked_initial_mask_decode(model, images, points, labels, 2)
    hook.remove()
    assert calls == []


def test_permutation_and_exact_chunk_call_counts_preserve_every_object():
    model, images, points, labels = fixture()
    sizes = []
    hook = model.prompt_encoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: sizes.append(kwargs['input_points'].shape[1]),
        with_kwargs=True)
    with torch.inference_mode():
        original = module.chunked_initial_mask_decode(model, images, points, labels, 3)
        order = [3, 1, 0, 2]
        permuted = module.chunked_initial_mask_decode(
            model, images, points[:, order], labels[:, order], 3)
    hook.remove()
    assert sizes == [3, 1, 3, 1]
    assert torch.allclose(permuted[0], original[0][order], atol=1e-4, rtol=0)
    assert torch.equal(permuted[1], original[1][order])


def test_source_inputs_state_flags_and_rng_are_unchanged():
    model, images, points, labels = fixture()
    inputs = [a.clone() for a in [*images, points, labels]]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        module.chunked_initial_mask_decode(model, images, points, labels, 3)
    assert all(torch.equal(a, b) for a, b in zip(inputs, [*images, points, labels], strict=True))
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('kind', ['training', 'trainable', 'gradient'])
def test_frozen_model_contract(kind):
    model, images, points, labels = fixture()
    if kind == 'training':
        model.train()
    elif kind == 'trainable':
        next(model.parameters()).requires_grad_(True)
    else:
        p = next(model.parameters())
        p.grad = torch.ones_like(p)
    with pytest.raises(ValueError):
        module.chunked_initial_mask_decode(model, images, points, labels, 2)
