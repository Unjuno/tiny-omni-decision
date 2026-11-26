import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from transformers.models.sam2_video.configuration_sam2_video import (  # noqa: E402
    Sam2VideoMaskDecoderConfig,
    Sam2VideoPromptEncoderConfig,
)
from transformers.models.sam2_video.modeling_sam2_video import (  # noqa: E402
    Sam2VideoMaskDecoder,
    Sam2VideoModel,
    Sam2VideoPositionalEmbedding,
    Sam2VideoPromptEncoder,
)

from tiny_omni_decision import sam2_batch_masks as module  # noqa: E402


class TinyRealHeads(torch.nn.Module):
    """Actual installed native heads only; no vision backbone/checkpoint/GPU."""
    get_image_wide_positional_embeddings = Sam2VideoModel.get_image_wide_positional_embeddings
    _single_frame_forward = Sam2VideoModel._single_frame_forward

    def __init__(self):
        super().__init__()
        p = Sam2VideoPromptEncoderConfig(hidden_size=8, image_size=16, patch_size=4,
                                        mask_input_channels=4)
        d = Sam2VideoMaskDecoderConfig(hidden_size=8, mlp_dim=16, num_hidden_layers=1,
                                      num_attention_heads=2, iou_head_hidden_dim=8)
        d._attn_implementation = 'sdpa'
        self.prompt_encoder = Sam2VideoPromptEncoder(p)
        self.shared_image_embedding = Sam2VideoPositionalEmbedding(p)
        self.mask_decoder = Sam2VideoMaskDecoder(d)
        self.image_size = 16
        self.object_pointer_proj = torch.nn.Linear(8, 8)
        self.no_object_pointer = torch.nn.Parameter(torch.zeros(1, 8))


def fixture():
    torch.manual_seed(17)
    model = TinyRealHeads().eval().requires_grad_(False)
    # Exercise real foreground masks, not a vacuous all-NO_OBJ_SCORE comparison.
    with torch.no_grad():
        for parameter in model.mask_decoder.pred_obj_score_head.parameters():
            parameter.zero_()
        model.mask_decoder.pred_obj_score_head.proj_out.bias.fill_(1.)
    images = [torch.randn(1, 1, 16, 16), torch.randn(1, 2, 8, 8), torch.randn(1, 8, 4, 4)]
    points = torch.tensor([[[[2., 3.]], [[5., 6.]], [[8., 9.]], [[12., 14.]]]])
    labels = torch.ones((1, 4, 1), dtype=torch.int64)
    return model, images, points, labels


def test_batched_real_heads_match_native_serial_low_resolution_semantics():
    model, images, points, labels = fixture()
    with torch.inference_mode():
        expected = [model._single_frame_forward(
            image_embeddings=images, input_points=points[:, i:i + 1],
            input_labels=labels[:, i:i + 1], multimask_output=True) for i in range(4)]
        raw = torch.cat([o.pred_masks for o in expected], 0)[:, 0]
        score = torch.cat([o.object_score_logits for o in expected], 0).flatten()
        actual, actual_score = module.initial_mask_decode(model, images, points, labels)
    assert (raw != -1024.).any()
    assert actual.shape == (4, 16, 16) and actual_score.shape == (4,)
    assert torch.allclose(actual, raw, atol=1e-4, rtol=0)
    assert torch.allclose(actual_score, score, atol=1e-5, rtol=0)
    assert torch.equal(actual > 0, raw > 0)


def test_direct_head_inputs_and_frozen_parameters_are_not_mutated():
    model, images, points, labels = fixture()
    originals = [a.clone() for a in [*images, points, labels]]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        module.initial_mask_decode(model, images, points, labels)
    assert all(torch.equal(a, b) for a, b in zip([*images, points, labels], originals, strict=True))
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize('kind', ['multi_point', 'negative', 'float_labels', 'multi_image'])
def test_only_reference_independent_positive_point_contract_admitted(kind):
    model, images, points, labels = fixture()
    if kind == 'multi_point':
        points, labels = points.repeat(1, 1, 2, 1), labels.repeat(1, 1, 2)
    elif kind == 'negative':
        labels[0, 0, 0] = 0
    elif kind == 'float_labels':
        labels = labels.float()
    else:
        points, labels = points.repeat(2, 1, 1, 1), labels.repeat(2, 1, 1)
    with pytest.raises(ValueError):
        module.initial_mask_decode(model, images, points, labels)


def test_trainable_or_training_mode_not_admitted_to_frozen_probe():
    model, images, points, labels = fixture()
    model.train()
    with pytest.raises(ValueError):
        module.initial_mask_decode(model, images, points, labels)
    model.eval().requires_grad_(True)
    with pytest.raises(ValueError):
        module.initial_mask_decode(model, images, points, labels)
