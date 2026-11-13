"""Independent-frame batched SAM2 low-resolution outputs; no tracking memory."""


def initial_mask_decode(model, image_embeddings, input_points, input_labels):
    """Native frozen prompt/decoder heads for one positive point per object.

    Embeddings must already have the native initial-frame no-memory addition.
    Only low-res masks/presence are returned; no high-resolution interpolation,
    object-pointer projection or temporal-memory encoding is needed for proposals.
    The real heads run as one image and many independent object prompts.
    """
    import torch

    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('frozen eval model with no gradients required')
    if (
        not isinstance(input_points, torch.Tensor) or input_points.dtype != torch.float32
        or input_points.ndim != 4 or input_points.shape[0] != 1
        or input_points.shape[2:] != (1, 2) or input_points.shape[1] < 1
        or not isinstance(input_labels, torch.Tensor)
        or input_labels.dtype not in (torch.int32, torch.int64)
        or input_labels.shape != input_points.shape[:3]
        or not torch.isfinite(input_points).all() or not (input_labels == 1).all()
        or input_labels.device != input_points.device
        or len(image_embeddings) != 3
    ):
        raise ValueError('one-image finite FP32 independent positive-point prompts required')
    for feature in image_embeddings:
        if (not isinstance(feature, torch.Tensor) or feature.ndim != 4 or feature.shape[0] != 1
                or feature.dtype != torch.float32 or feature.device != input_points.device
                or not torch.isfinite(feature).all()):
            raise ValueError('three same-frame finite FP32 image feature maps required')
    sparse, dense = model.prompt_encoder(
        input_points=input_points, input_labels=input_labels, input_boxes=None, input_masks=None)
    raw, iou, _, presence = model.mask_decoder(
        image_embeddings=image_embeddings[-1],
        image_positional_embeddings=model.get_image_wide_positional_embeddings(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=True, high_resolution_features=image_embeddings[:-1])
    return select_sam2_masks(raw, iou, presence)


def select_sam2_masks(mask_logits, iou_scores, object_scores):
    """Keep every positive-point object; native best-IoU and no-object semantics.

    One image, independent object axis. Correct per-object gather avoids the
    cross-batch advanced-index broadcast in the full high-resolution wrapper.
    No extra score gate, mask threshold, NMS or candidate filtering is applied.
    """
    import torch

    values = (mask_logits, iou_scores, object_scores)
    if (
        any(not isinstance(a, torch.Tensor) or a.dtype != torch.float32 for a in values)
        or mask_logits.ndim != 5 or mask_logits.shape[0] != 1
        or min(mask_logits.shape) == 0
        or iou_scores.shape != mask_logits.shape[:3]
        or object_scores.shape != mask_logits.shape[:2] + (1,)
        or any(a.device != mask_logits.device or not torch.isfinite(a).all() for a in values)
    ):
        raise ValueError('finite FP32 one-image aligned multimask/IoU/presence tensors required')
    _, q, _, h, w = mask_logits.shape
    best = iou_scores.argmax(-1)
    selected = mask_logits.gather(2, best[..., None, None, None].expand(1, q, 1, h, w))
    selected = selected[0, :, 0]
    present = object_scores[0, :, 0] > 0
    selected = torch.where(present[:, None, None], selected, -1024.)
    return selected, object_scores[0, :, 0]
