"""Compare invertible sensor representations within frozen DROID train groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import OPTIONS, SOURCE_COMMIT, TRAIN_SHA, digest

SUMMARY_SHA = '4a764517ed01c0ac0b9727d53b0287734fa9e4325d0937ce58fb91f7edfb221f'
SENSOR_SHA = 'de8dc6de9afec85932b31d08d94ed58d097896e03ad1098813fd9f2c7ce372b9'
LABEL_SHA = '372579d3a5b639aa0170684abc9863b24d10ab3dee6e3821a3a261817fab80b1'
GRID = (100.0, 1000.0, 10000.0)


def temporal_features(sensor):
    x = np.asarray(sensor, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (8, 14) or not np.isfinite(x).all():
        raise ValueError('expected finite [N,8,14] sensor sequences')
    return {
        'raw': x.reshape(len(x), 112),
        'delta': np.concatenate((x[:, -1], np.diff(x, axis=1).reshape(len(x), 98)), axis=1),
    }


def nested_compare(arms, targets, groups, api):
    y = np.asarray(targets, dtype=int)
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape or not arms:
        raise ValueError('unaligned targets/groups/arms')
    if any(x.shape != (len(y), next(iter(arms.values())).shape[1]) for x in arms.values()):
        raise ValueError('arms must have equal aligned dimensions')
    folds = api['stratified_group_folds'](y, groups, folds=5, seed=17)
    if not np.array_equal(np.sort(np.concatenate(folds)), np.arange(len(y))):
        raise ValueError('outer folds must partition each row exactly once')
    probabilities = {name: np.zeros((len(y), 7)) for name in arms}
    prior = np.zeros((len(y), 7))
    records, models = [], {}
    for fold_id, held in enumerate(folds):
        train = np.ones(len(y), dtype=bool)
        train[held] = False
        if set(groups[train]) & set(groups[held]):
            raise ValueError('outer group leakage')
        prior[held] = np.bincount(y[train], minlength=7) / train.sum()
        selections = {}
        for name, x in arms.items():
            selection = api['select_l2_strength_group_cv'](
                x[train], y[train], groups[train].tolist(), candidates=GRID,
                folds=4, seed=17, max_iterations=2000,
            )
            model = api['fit_l2_softmax'](
                x[train], y[train], l2_strength=selection['selected_l2_strength'],
                max_iterations=2000,
            )
            probabilities[name][held] = api['predict_softmax'](model, x[held])
            models[f'{name}-outer-{fold_id}'] = model
            selections[name] = selection
            print(json.dumps({'outer_fold': fold_id, 'arm': name,
                              'selected_l2': selection['selected_l2_strength']}), flush=True)
        records.append({'outer_fold': fold_id, 'held_indices': held.tolist(),
                        'train_indices': np.flatnonzero(train).tolist(),
                        'selections': selections})
    return {'probabilities': probabilities, 'prior_probabilities': prior,
            'models': models, 'folds': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {args.train: TRAIN_SHA,
              args.feature_cache / 'summary.json': SUMMARY_SHA,
              args.feature_cache / 'sensor_sequences.npy': SENSOR_SHA,
              args.feature_cache / 'labels.npy': LABEL_SHA}
    if any(digest(path) != expected for path, expected in inputs.items()):
        raise ValueError('pinned input hash mismatch')
    rows = [json.loads(line) for line in args.train.read_text().splitlines()]
    if (len(rows) != 275 or any(r['split'] != 'train' for r in rows)
            or len({r['group_id'] for r in rows}) != len(rows)
            or any(tuple(r['options']) != OPTIONS for r in rows)):
        raise ValueError('unexpected frozen train manifest')
    sensor_map = np.load(args.feature_cache / 'sensor_sequences.npy', mmap_mode='r')
    label_map = np.load(args.feature_cache / 'labels.npy', mmap_mode='r')
    if sensor_map.shape != (425, 8, 14) or label_map.shape != (425,):
        raise ValueError('unexpected original feature shapes')
    # Pinned extractor read_rows() stores the original train rows before validation.
    x = np.array(sensor_map[:275], dtype=np.float64, copy=True)
    y = np.array(label_map[:275], dtype=int, copy=True)
    del sensor_map, label_map
    if not np.array_equal(y, [OPTIONS.index(r['target_option_id']) for r in rows]):
        raise ValueError('train feature/target alignment mismatch')
    code = subprocess.check_output([
        'git', 'show', f'{SOURCE_COMMIT}:src/tiny_omni_decision/droid_next_action_readout.py',
    ])
    extractor = subprocess.check_output([
        'git', 'show', f'{SOURCE_COMMIT}:scripts/extract_droid_fresh_video_sensor_features.py',
    ])
    api = {'__name__': 'pinned_readout'}
    exec(compile(code, 'pinned_readout', 'exec'), api)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'pinned-readout.py').write_bytes(code)
    (args.output / 'pinned-extractor.py').write_bytes(extractor)
    import scipy

    protocol = {
        'started_utc': datetime.now(UTC).isoformat(), 'source_commit': SOURCE_COMMIT,
        'script_sha256': digest(__file__),
        'plan_sha256': digest(Path('docs/superpowers/plans/2026-10-27-droid-temporal-train-cv.md')),
        'readout_sha256': hashlib.sha256(code).hexdigest(),
        'extractor_sha256': hashlib.sha256(extractor).hexdigest(),
        'input_hashes': {str(path): value for path, value in inputs.items()},
        'arms': ['raw', 'delta'], 'input_width': 112, 'parameters_per_head': 791,
        'outer_folds': 5, 'inner_folds': 4, 'seed': 17, 'l2_grid': GRID,
        'max_iterations': 2000, 'selection_metric': 'inner out-of-fold macro NLL',
        'primary_comparison': 'pooled outer macro NLL, delta minus raw',
        'train_rows': 275, 'held_development_validation_rows_accessed': 0,
        'caveat': 'whole containers hashed, only train rows accessed; not blind evaluation',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'platform': platform.platform()},
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.perf_counter()
    compared = nested_compare(temporal_features(x), y, [r['group_id'] for r in rows], api)
    measure = api['multiclass_probability_metrics']
    metrics = {name: measure(y, p) for name, p in compared['probabilities'].items()}
    metrics['outer_train_prior'] = measure(y, compared['prior_probabilities'])
    for fold in compared['folds']:
        held = np.asarray(fold['held_indices'])
        fold['held_group_ids'] = [rows[i]['group_id'] for i in held]
        fold['outer_metrics'] = {
            name: measure(y[held], p[held]) for name, p in compared['probabilities'].items()
        }
        train = np.asarray(fold['train_indices'])
        inner = api['stratified_group_folds'](
            y[train], [rows[i]['group_id'] for i in train], folds=4, seed=17)
        fold['inner_held_indices_in_full_train'] = [train[index].tolist() for index in inner]
    for name, model in compared['models'].items():
        np.savez(args.output / f'{name}.npz', **model)
    prediction_path = args.output / 'outer-train-predictions.jsonl'
    prediction_path.write_text(''.join(json.dumps({
        'id': row['id'], 'group_id': row['group_id'], 'target': int(y[i]),
        'probabilities': {name: p[i].tolist() for name, p in compared['probabilities'].items()},
    }) + '\n' for i, row in enumerate(rows)))
    if any(digest(path) != expected for path, expected in inputs.items()):
        raise ValueError('pinned inputs changed during comparison')
    result = {**protocol, 'completed_utc': datetime.now(UTC).isoformat(),
              'wall_seconds': time.perf_counter() - started, 'metrics': metrics,
              'folds': compared['folds'], 'predictions_sha256': digest(prediction_path),
              'model_hashes': {p.name: digest(p) for p in sorted(args.output.glob('*.npz'))},
              'input_hashes_unchanged': True, 'gpu_or_encoder_used': False,
              'sealed_audit_loaded': False, 'product_head_created': False}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'metrics': metrics, 'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
