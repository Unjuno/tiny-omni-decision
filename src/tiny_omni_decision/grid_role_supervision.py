"""Weak bbox supervision for immutable native grid inputs; never runtime IDs."""

import numpy as np

from tiny_omni_decision.proposal_opportunity import valid_boxes


def grid_cells(size):
    """Row-major normalized cell extents, matching whole-grid mean pooling."""
    if type(size) is not int or size < 1:
        raise ValueError("positive integer grid size required")
    return np.array(
        [
            [x / size, y / size, (x + 1) / size, (y + 1) / size]
            for y in range(size)
            for x in range(size)
        ],
        dtype=np.float64,
    )


def cell_source_labels(cells, boxes_by_id):
    """Whole-cell containment and no positive overlap with another available role.

    Missing competitor annotations and bbox background remain limitations. This
    does not establish foreground membership, visibility or physical identity.
    Unknown=-1, competing-role ambiguity=-2; neither is a negative training label.
    """
    if not isinstance(cells, np.ndarray) or cells.dtype != np.float64:
        raise ValueError("native FP64 cell geometry required")
    cells = valid_boxes(cells, positive=True)
    if not isinstance(boxes_by_id, dict) or any(type(i) is not int or i < 0 for i in boxes_by_id):
        raise ValueError("unique nonnegative source IDs required")
    labels = np.full(len(cells), -1, np.int64)
    if boxes_by_id:
        ids = np.array(list(boxes_by_id), np.int64)
        boxes = valid_boxes(np.array(list(boxes_by_id.values())), positive=True)
        contains = (cells[:, None, :2] >= boxes[None, :, :2]).all(-1) & (
            cells[:, None, 2:] <= boxes[None, :, 2:]
        ).all(-1)
        overlap = (
            np.minimum(cells[:, None, 2:], boxes[None, :, 2:])
            > np.maximum(cells[:, None, :2], boxes[None, :, :2])
        ).all(-1)
        unique = (contains.sum(1) == 1) & (overlap.sum(1) == 1)
        labels[unique] = ids[contains[unique].argmax(1)]
        labels[contains.any(1) & ~unique] = -2
    return labels, dict(
        total=len(labels),
        unique=int((labels >= 0).sum()),
        unknown=int((labels == -1).sum()),
        ambiguous=int((labels == -2).sum()),
    )


def pair_role_counts(labels, pairs, *, hidden, group, cover):
    """Report actual role support without selecting/duplicating any pair."""
    if (
        type(hidden) is not int
        or type(cover) is not int
        or hidden < 0
        or not isinstance(group, list)
        or not group
        or any(type(i) is not int or i < 0 for i in group)
        or len(set(group)) != len(group)
        or hidden in group
        or cover not in group
    ):
        raise ValueError("distinct grounded hidden/group/cover namespaces required")
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.dtype != np.int64
        or not isinstance(pairs, np.ndarray)
        or pairs.dtype != np.int64
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or np.any(pairs < 0)
        or np.any(pairs >= len(labels))
    ):
        raise ValueError("aligned integer labels and bounded pairs required")
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    allowed = group + [hidden]
    if not np.isin(a, allowed).all() or not np.isin(b, allowed).all():
        raise ValueError("pair endpoints must have eligible task-role labels")
    same = a == b
    counts = dict(
        total=len(pairs),
        same_hidden=int((same & (a == hidden)).sum()),
        same_group=int((same & np.isin(a, group)).sum()),
        different_hidden_group=int((~same & ((a == hidden) | (b == hidden))).sum()),
        different_group_group=int((~same & (a != hidden) & (b != hidden)).sum()),
        same_cover=int((same & (a == cover)).sum()),
    )
    if (
        sum(
            counts[k]
            for k in (
                "same_hidden",
                "same_group",
                "different_hidden_group",
                "different_group_group",
            )
        )
        != counts["total"]
    ):
        raise ValueError("role support partition incomplete")
    return counts
