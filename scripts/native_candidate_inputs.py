"""Small deterministic contracts for the native candidate integration smoke."""

import math
import subprocess

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


def require_source_revision(source, revision, package):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if git("rev-parse", "HEAD") != revision or git("diff", "HEAD", "--name-only"):
        raise ValueError("external source HEAD/index/worktree differs")
    extra = git("ls-files", "--others", "--exclude-standard", "--", package).splitlines()
    if any(p.endswith((".py", ".pyd", ".dll")) for p in extra):
        raise ValueError("untracked importable model source")


def compare_decisions(current, reference, *, atol):
    if len(current) != len(reference):
        raise ValueError("replay decision count differs")
    maximum = 0.0
    for a, b in zip(current, reference, strict=True):
        if any(a[k] != b[k] for k in ["modality", "question", "candidates"]):
            raise ValueError("replay candidate/condition association differs")
        a, b = a["decision"], b["decision"]
        if (a["selected_option_id"] != b["selected_option_id"]
                or a["probabilities"].keys() != b["probabilities"].keys()):
            raise ValueError("replay selected ID or probability keys differ")
        for key, value in a["probabilities"].items():
            expected = b["probabilities"][key]
            if not math.isfinite(value) or not math.isfinite(expected):
                raise ValueError("nonfinite replay probability")
            maximum = max(maximum, abs(value - expected))
    if maximum > atol:
        raise ValueError("replay probability tolerance exceeded")
    return maximum
