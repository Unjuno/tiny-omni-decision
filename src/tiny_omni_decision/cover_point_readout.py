"""Input-only cover membership; source boxes are used only for supervision."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def initial_cover_membership(source, query_xy):
    """Return labels for every query; refuse missing/ambiguous frame-zero targets."""
    questions = [
        q
        for q in source.get("grounded_question", [])
        if q["question"].endswith("Track the object that occludes the smaller object.")
    ]
    if len(questions) != 1 or len(questions[0]["answers"]) != 1:
        raise ValueError("one grounded cover required")
    tracks = [t for t in source["object_tracking"] if t["id"] == questions[0]["answers"][0]]
    if len(tracks) != 1 or tracks[0]["frame_ids"].count(0) != 1:
        raise ValueError("one frame-zero cover box required")
    box = np.asarray(tracks[0]["bounding_boxes"][tracks[0]["frame_ids"].index(0)])
    xy = np.asarray(query_xy)
    if np.iscomplexobj(box) or np.iscomplexobj(xy):
        raise ValueError("real coordinates required")
    box, xy = box.astype(np.float64), xy.astype(np.float64)
    if (
        box.shape != (4,)
        or xy.ndim != 2
        or xy.shape[1] != 2
        or len(xy) == 0
        or not np.isfinite(box).all()
        or not np.isfinite(xy).all()
        or (box[:2] >= box[2:]).any()
        or (box < 0).any()
        or (box > 1).any()
        or (xy < 0).any()
        or (xy > 1).any()
    ):
        raise ValueError("nonempty normalized finite coordinates and positive-area box required")
    return ((xy >= box[:2]) & (xy <= box[2:])).all(1)


class CoverPointReadout(nn.Module):
    """Equivariant membership logits using local tokens and full scene context."""

    def __init__(self, input_dim=512, hidden=64):
        super().__init__()
        self.input_dim = input_dim
        self.point = nn.Linear(input_dim, hidden)
        self.context = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1)

    def encode(self, points, mask=None):
        """Return shared point representations and their validated padding mask."""
        if (
            points.ndim != 3
            or points.shape[-1] != self.input_dim
            or min(points.shape[:2]) == 0
            or not torch.isfinite(points).all()
        ):
            raise ValueError("nonempty finite aligned point tokens required")
        if mask is None:
            mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if mask.shape != points.shape[:2] or mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError("aligned Boolean mask with at least one point per clip required")
        local = self.point(points).tanh()
        scene = (local * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        hidden = (local + self.context(scene)[:, None]).tanh()
        return hidden, mask

    def forward(self, points, mask=None):
        hidden, mask = self.encode(points, mask)
        return self.score(hidden).squeeze(-1).masked_fill(~mask, -torch.inf)


def balanced_membership_loss(logits, targets):
    """Equal clips/classes; retain all-negative and all-positive rows."""
    if (
        logits.ndim != 2
        or min(logits.shape) == 0
        or targets.shape != logits.shape
        or targets.dtype != torch.bool
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("nonempty finite logits and aligned Boolean labels required")
    loss = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    positive, negative = targets.sum(1), (~targets).sum(1)
    pos = (loss * targets).sum(1) / positive.clamp_min(1)
    neg = (loss * ~targets).sum(1) / negative.clamp_min(1)
    classes = (positive > 0).to(logits.dtype) + (negative > 0).to(logits.dtype)
    return ((pos + neg) / classes).mean()
