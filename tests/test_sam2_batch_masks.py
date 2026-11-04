import pytest

torch = pytest.importorskip('torch')

from tiny_omni_decision import sam2_batch_masks as module  # noqa: E402


def inputs():
    masks = torch.arange(1 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 3, 3, 2, 2)
    iou = torch.tensor([[[.1, .8, .3], [.9, .8, .1], [.1, .2, .8]]])
    presence = torch.tensor([[[2.], [0.], [-1.]]])
    return masks, iou, presence


def test_per_object_gather_avoids_cross_batch_mix_and_matches_presence_gate():
    masks, iou, presence = inputs()
    chosen, scores = module.select_sam2_masks(masks, iou, presence)
    assert chosen.shape == (3, 2, 2) and scores.shape == (3,)
    assert chosen.dtype == scores.dtype == torch.float32
    assert torch.equal(chosen[0], torch.tensor([[4., 5], [6, 7]]))
    assert torch.equal(chosen[1:], torch.full((2, 2, 2), -1024., dtype=torch.float32))
    assert torch.equal(scores, torch.tensor([2., 0., -1.]))


def test_iou_ties_keep_first_multimask_without_filtering_objects():
    masks, iou, presence = inputs()
    iou[:] = .5
    presence[:] = 1
    before = masks.clone()
    chosen, scores = module.select_sam2_masks(masks, iou, presence)
    assert torch.equal(chosen, masks[0, :, 0])
    assert torch.equal(scores, torch.ones(3))
    assert torch.equal(masks, before)


def test_object_permutation_preserves_candidate_order():
    masks, iou, presence = inputs()
    original = module.select_sam2_masks(masks, iou, presence)
    order = [2, 0, 1]
    permuted = module.select_sam2_masks(masks[:, order], iou[:, order], presence[:, order])
    assert torch.equal(permuted[0], original[0][order])
    assert torch.equal(permuted[1], original[1][order])


@pytest.mark.parametrize('index', [0, 1, 2])
def test_nonfinite_decoder_outputs_fail_closed(index):
    data = list(inputs())
    data[index].flatten()[0] = torch.nan
    with pytest.raises(ValueError):
        module.select_sam2_masks(*data)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_only_exact_fp32_reference_dtype_admitted(index):
    data = list(inputs())
    data[index] = data[index].to(torch.float16)
    with pytest.raises(ValueError):
        module.select_sam2_masks(*data)


@pytest.mark.parametrize('index,shape', [(0, (2, 3, 3, 2, 2)), (0, (1, 0, 3, 2, 2)),
                                       (1, (1, 3)), (1, (1, 4, 3)), (2, (1, 3)),
                                       (2, (1, 2, 1))])
def test_single_image_aligned_multimask_schema_required(index, shape):
    data = list(inputs())
    data[index] = torch.ones(shape, dtype=torch.float32)
    with pytest.raises(ValueError):
        module.select_sam2_masks(*data)
