"""Train-asset-only PCA and complete-coverage gates for the fixed audio comparison."""

import numpy as np

from scripts.clap_full_clip import duration_weighted_feature, window_spans


def fit_train_projection(features, source_ids, train_ids, dimensions):
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2 or len(values) != len(source_ids)
        or len(set(source_ids)) != len(source_ids)
        or len(set(train_ids)) != len(train_ids)
        or not set(train_ids).issubset(source_ids)
        or not isinstance(dimensions, int) or isinstance(dimensions, bool)
        or dimensions < 1 or dimensions >= values.shape[1]
        or len(train_ids) <= dimensions or not np.isfinite(values).all()
    ):
        raise ValueError('invalid unique train identities or projection capacity')
    lookup = {identity: i for i, identity in enumerate(source_ids)}
    train = values[[lookup[identity] for identity in train_ids]]
    mean = train.mean(0)
    _, singular, components = np.linalg.svd(train - mean, full_matrices=False)
    if singular[dimensions - 1] <= np.finfo(float).eps * max(train.shape) * singular[0]:
        raise ValueError('training features have insufficient projection rank')
    components = components[:dimensions].copy()
    # Fix arbitrary SVD signs without looking at labels or held-out features.
    peaks = np.abs(components).argmax(1)
    components *= np.sign(components[np.arange(dimensions), peaks])[:, None]
    return dict(mean=mean, components=components, singular_values=singular,
                fit_ids=list(train_ids),
                retained_variance=float(np.square(singular[:dimensions]).sum()
                                        / np.square(singular).sum()))


def project(features, projection):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('invalid features for fixed projection')
    result = (values - projection['mean']) @ projection['components'].T
    if not np.isfinite(result).all():
        raise ValueError('nonfinite projected features')
    return result.astype(np.float32)


def validate_coverage(records, report, pooled, windows):
    if (report['status'] != 'PASS' or report['scope'] != 'full'
            or report['repeat_bitwise_equal'] is not True):
        raise ValueError('full repeatable extraction required')
    rows = report['rows']
    if (len(rows) != len(records) or pooled.ndim != 2 or windows.ndim != 2
            or len(pooled) != len(records) or pooled.shape[1] != windows.shape[1]
            or not np.isfinite(pooled).all() or not np.isfinite(windows).all()):
        raise ValueError('feature coverage or shape invalid')
    offset = 0
    for record, row, feature in zip(records, rows, pooled, strict=True):
        spans = window_spans(row['samples'], report['window_samples'])
        if (row['id'] != record['source_id'] or row['split'] != record['split']
                or row['spans'] != [list(span) for span in spans]
                or row['covered_samples'] != row['samples']
                or row['window_offset'] != offset or row['windows'] != len(spans)):
            raise ValueError('source identity, split or temporal coverage differs')
        vectors = windows[offset:offset + len(spans)]
        expected = duration_weighted_feature(vectors, [b - a for a, b in spans])
        if not np.allclose(feature, expected, atol=1e-7, rtol=1e-6):
            raise ValueError('stored pooled feature differs from complete windows')
        offset += len(spans)
    if offset != len(windows):
        raise ValueError('extra or missing window features')
