"""Independent-frame SAM2 head chunks; keep every ordered positive-point object."""

from tiny_omni_decision.sam2_batch_masks import initial_mask_decode


def extend_pinned_sources(inherited, added):
    """Append new code pins while rejecting changes to protected source identity."""
    if any(key in inherited and inherited[key] != value for key, value in added.items()):
        raise ValueError('protected source pin cannot be changed')
    return {**inherited, **added}


def chunked_initial_mask_decode(model, image_embeddings, input_points, input_labels, chunk_size):
    """Use the unchanged native head helper in contiguous chunks, including tail.

    Same cached image features for every chunk; no temporal propagation or NMS.
    Validate the complete prompt set before executing any partial head forward.
    Chunk size is an explicit experiment input, never a post-hoc retry policy.
    """
    import torch

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive integer chunk size required')
    if (
        not isinstance(input_points, torch.Tensor) or input_points.dtype != torch.float32
        or input_points.ndim != 4 or input_points.shape[0] != 1
        or input_points.shape[2:] != (1, 2) or input_points.shape[1] < 1
        or not isinstance(input_labels, torch.Tensor)
        or input_labels.dtype not in (torch.int32, torch.int64)
        or input_labels.shape != input_points.shape[:3]
        or input_labels.device != input_points.device
        or not torch.isfinite(input_points).all() or not (input_labels == 1).all()
    ):
        raise ValueError('complete finite FP32 independent positive-point inventory required')
    chunks = [initial_mask_decode(model, image_embeddings,
                                  input_points[:, i:i + chunk_size],
                                  input_labels[:, i:i + chunk_size])
              for i in range(0, input_points.shape[1], chunk_size)]
    return torch.cat([a for a, _ in chunks], 0), torch.cat([b for _, b in chunks], 0)
