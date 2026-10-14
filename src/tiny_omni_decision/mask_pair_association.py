"""Annotation-conditioned correspondence diagnostics; no GT in input features."""

import numpy as np


def association_windows(count, *, gap=8):
    if type(count) is not int or type(gap) is not int or not 2 <= gap <= count - 2:
        raise ValueError("distinct adjacent, long-gap and endpoint windows required")
    return (
        [(i, i + 1, 0) for i in range(count - 1)]
        + [(i, i + gap, 1) for i in range(count - gap)]
        + [(0, count - 1, 2)]
    )


def labelled_mask_pairs(labels, offsets, windows):
    """Supervision-only eligibility; unknown/ambiguous pairs stay in the ledger."""
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.dtype != np.int64
        or np.any(labels < -2)
        or not isinstance(offsets, np.ndarray)
        or offsets.ndim != 1
        or offsets.dtype != np.int64
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(labels)
        or np.any(np.diff(offsets) < 0)
        or not isinstance(windows, list)
        or not windows
    ):
        raise ValueError("aligned integer supervision/frame offsets required")
    seen = set()
    for window in windows:
        if (
            not isinstance(window, tuple)
            or len(window) != 3
            or any(type(v) is not int for v in window)
            or not 0 <= window[0] < window[1] < len(offsets) - 1
            or window[2] not in (0, 1, 2)
            or window in seen
        ):
            raise ValueError("unique forward-time windows with valid strata required")
        seen.add(window)
    result, targets, strata = [], [], []
    ledger = dict(
        all_pairs=0,
        labelled_pairs=0,
        unknown_or_ambiguous_pairs=0,
        unknown_pairs=0,
        ambiguous_pairs=0,
        unknown_and_ambiguous_pairs=0,
    )
    strata_ledger = {str(k): dict(ledger) for k in range(3)}
    for first, last, stratum in windows:
        a, b = np.meshgrid(
            np.arange(offsets[first], offsets[first + 1], dtype=np.int64),
            np.arange(offsets[last], offsets[last + 1], dtype=np.int64),
            indexing="ij",
        )
        a, b = a.ravel(), b.ravel()
        unknown = (labels[a] == -1) | (labels[b] == -1)
        ambiguous = (labels[a] == -2) | (labels[b] == -2)
        eligible = (labels[a] >= 0) & (labels[b] >= 0)
        values = dict(
            all_pairs=len(a),
            labelled_pairs=int(eligible.sum()),
            unknown_or_ambiguous_pairs=int((~eligible).sum()),
            unknown_pairs=int(unknown.sum()),
            ambiguous_pairs=int(ambiguous.sum()),
            unknown_and_ambiguous_pairs=int((unknown & ambiguous).sum()),
        )
        for key, value in values.items():
            ledger[key] += value
            strata_ledger[str(stratum)][key] += value
        result.append(np.column_stack((a[eligible], b[eligible])))
        targets.append((labels[a[eligible]] == labels[b[eligible]]).astype(np.int64))
        strata.append(np.full(int(eligible.sum()), stratum, np.int64))
    ledger["strata"] = strata_ledger
    return np.concatenate(result), np.concatenate(targets), np.concatenate(strata), ledger


def grouped_pair_split(video_indices, held_out):
    if (
        not isinstance(video_indices, np.ndarray)
        or video_indices.ndim != 1
        or video_indices.dtype != np.int64
        or np.any(video_indices < 0)
        or type(held_out) is not int
        or held_out < 0
    ):
        raise ValueError("integer scene membership required")
    test = video_indices == held_out
    if not test.any() or test.all():
        raise ValueError("nonempty independent scene training and holdout pools required")
    return ~test, test


def pair_features(appearance, boxes, times, pairs, clip_frames):
    """Symmetric appearance difference/product and five model-geometry features."""
    if (
        not isinstance(appearance, np.ndarray)
        or appearance.ndim != 2
        or min(appearance.shape) < 1
        or appearance.dtype != np.float32
        or not np.isfinite(appearance).all()
        or not isinstance(boxes, np.ndarray)
        or boxes.shape != (len(appearance), 4)
        or boxes.dtype != np.float64
        or not np.isfinite(boxes).all()
        or np.any(boxes < 0)
        or np.any(boxes > 1)
        or not isinstance(times, np.ndarray)
        or times.shape != (len(appearance),)
        or times.dtype != np.int64
        or type(clip_frames) is not int
        or clip_frames < 2
        or np.any(times < 0)
        or np.any(times >= clip_frames)
        or not isinstance(pairs, np.ndarray)
        or pairs.ndim != 2
        or pairs.shape[1] != 2
        or pairs.dtype != np.int64
        or np.any(pairs < 0)
        or np.any(pairs >= len(appearance))
    ):
        raise ValueError(
            "finite aligned FP32 appearance, native FP64 boxes and pair indices required"
        )
    a, b = pairs[:, 0], pairs[:, 1]
    size = boxes[:, 2:] - boxes[:, :2]
    if np.any(size[a] <= 0) or np.any(size[b] <= 0):
        raise ValueError("paired model boxes must have positive extent")
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    geometry = np.concatenate(
        (
            np.abs(center[a] - center[b]),
            np.abs(np.log(size[a]) - np.log(size[b])),
            (np.abs(times[a] - times[b]) / (clip_frames - 1))[:, None],
        ),
        1,
    )
    av, bv = appearance[a].astype(np.float64), appearance[b].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.concatenate((np.abs(av - bv), av * bv, geometry), 1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("pair features exceed finite FP32")
    return result


def training_standardizer(features, training_mask):
    if (
        not isinstance(features, np.ndarray)
        or features.ndim != 2
        or min(features.shape) < 1
        or features.dtype != np.float32
        or not np.isfinite(features).all()
        or not isinstance(training_mask, np.ndarray)
        or training_mask.dtype != np.bool_
        or training_mask.shape != (len(features),)
        or not training_mask.any()
    ):
        raise ValueError("finite pair features and nonempty training-only mask required")
    values = features[training_mask].astype(np.float64)
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 1e-6).astype(np.float32)


def paired_scene_interval(differences):
    if (
        not isinstance(differences, np.ndarray)
        or differences.ndim != 1
        or len(differences) < 2
        or differences.dtype.kind != "f"
        or not np.isfinite(differences).all()
    ):
        raise ValueError("finite scene-mean differences from at least two groups required")
    sampled = np.random.default_rng(17).integers(0, len(differences), (2000, len(differences)))
    means = differences[sampled].astype(np.float64).mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(
        mean=float(differences.mean()),
        low=float(low),
        high=float(high),
        groups=len(differences),
        replicates=2000,
        seed=17,
    )
