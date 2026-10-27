"""Post-hoc connected-component paired bootstrap of fixed DROID predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import digest

SOURCE = Path('C:/CodexArtifacts/pretrained-reuse-r2/droid-prior-init-20261028-attempt-01/oof.npz')
SOURCE_SHA = 'e2538c07a56687c08c5eb7cc22ab4376613700cd2e21b450ad73a799b28aed7d'
REPO = Path(__file__).resolve().parents[1]


def paired_deltas(labels, model, prior):
    row = np.arange(len(labels))
    acc = (model.argmax(axis=1) == labels).astype(float)
    acc -= (prior.argmax(axis=1) == labels).astype(float)
    nll = -np.log(np.maximum(model[row, labels], 1e-12))
    nll += np.log(np.maximum(prior[row, labels], 1e-12))
    return np.column_stack((acc, nll))


def bootstrap(deltas, groups, repeats=5000, seed=17):
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2 or len(deltas) != len(groups):
        raise ValueError('invalid groups')
    sums = np.zeros((len(unique), deltas.shape[1]))
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, deltas)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(unique), (repeats, len(unique)))
    values = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)[:, None]
    return np.quantile(values, [0.025, 0.975], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists() or output.is_relative_to(REPO) or digest(SOURCE) != SOURCE_SHA:
        raise ValueError('output or source hash invalid')
    data = np.load(SOURCE)
    y, groups = data['labels'], data['groups']
    model, prior = data['sensor'], data['sensor_video']
    if len(y) != 275 or len(np.unique(groups)) != 268:
        raise ValueError('unexpected development cohort')
    deltas = paired_deltas(y, model, prior)
    interval = bootstrap(deltas, groups)
    result = {'method': 'paired connected-component bootstrap; percentile 95%',
              'seed': 17, 'replicates': 5000, 'components': 268, 'rows': 275,
              'sensor_minus_prior': {'accuracy_delta': float(deltas[:, 0].mean()),
                                     'nll_delta': float(deltas[:, 1].mean()),
                                     'accuracy_interval': interval[:, 0].tolist(),
                                     'nll_interval': interval[:, 1].tolist()},
              'source_sha256': SOURCE_SHA, 'script_sha256': digest(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
