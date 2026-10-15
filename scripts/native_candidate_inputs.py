"""Small deterministic contracts for the native candidate integration smoke."""

import torch


def positions(kind, count):
    """Relative sampled-order/grid coordinates, not cross-source synchronization."""
    if kind in {"text", "speech", "event_audio"} and type(count) is int and count > 0:
        result = torch.zeros(count, 3)
        result[:, 0] = torch.linspace(0, 1, count)
        return result
    if kind == "image" and count == 17:
        grid = torch.cartesian_prod(torch.zeros(1), torch.linspace(0, 1, 4),
                                    torch.linspace(0, 1, 4))
        return torch.cat([torch.tensor([[0., .5, .5]]), grid])
    if kind == "video" and count == 2048:
        return torch.cartesian_prod(torch.linspace(0, 1, 8), torch.linspace(0, 1, 16),
                                    torch.linspace(0, 1, 16))
    raise ValueError("declared native token profile required")


def validate_native_sources(images, audiovisual, *, image_id, image_sha,
                            source_id, audio_sha, video_sha):
    images = [v for v in images.get("train", []) if v["id"] == image_id]
    av = [v for v in audiovisual if v["source_id"] == source_id]
    if (len(images) != 1 or images[0].get("split") != "train"
            or images[0]["sha256"] != image_sha or len(av) != 1
            or av[0].get("split") != "train" or av[0]["audio_sha256"] != audio_sha
            or av[0]["video_sha256"] != video_sha
            or av[0]["audio_status"] != "PASS" or av[0]["video_status"] != "PASS"):
        raise ValueError("hash-matched declared TRAIN image/audio/video required")


def masked_mean(hidden, mask):
    if (hidden.ndim != 3 or mask.shape != hidden.shape[:2]
            or not torch.isfinite(hidden).all() or not torch.isfinite(mask).all()
            or not ((mask == 0) | (mask == 1)).all() or not (mask.sum(1) > 0).all()):
        raise ValueError("finite aligned language tokens and nonempty binary masks required")
    return (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
