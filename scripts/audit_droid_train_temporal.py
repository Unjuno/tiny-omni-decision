"""Train-only deterministic diagnostics; no fitting or validation scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SOURCE_COMMIT = 'eba818067c6a9ef9089288205c26f6a8254ca6e5'
RECEIPT_SHA = '5d5a5ba5d59b8c76790db1c9399f6490e70098ad390efafe5ef7f94df63ee6f7'
TRAIN_SHA = '4e14850bbbe2eee2a96116a97ca935dfd5dea91d197c00639ea7ed02e2d93341'
OPTIONS = ('stop', 'x-', 'x+', 'y-', 'y+', 'z-', 'z+')


def direction(vector):
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('expected three finite translation values')
    if np.max(np.abs(values)) <= 1e-4:
        return 0
    axis = int(np.argmax(np.abs(values)))
    return 1 + 2 * axis + int(values[axis] > 0)


def temporal_predictions(positions, actions, indices, target):
    indices = list(indices)
    if (len(indices) < 2 or indices != sorted(set(indices)) or indices[0] < 0
            or indices[-1] != target - 1 or indices[-1] < 1):
        raise ValueError('need strictly ordered past indices ending immediately before target')
    k = indices[-1]
    positions, actions = np.asarray(positions), np.asarray(actions)
    if (positions.ndim != 2 or actions.ndim != 2 or positions.shape[1] != 6
            or actions.shape[1] != 6 or min(len(positions), len(actions)) <= k):
        raise ValueError('expected observed position/action arrays with six columns')
    return {
        'previous_command': direction(actions[k, :3]),
        'sparse_displacement': direction(positions[k, :3] - positions[indices[-2], :3]),
        'dense_displacement': direction(positions[k, :3] - positions[k - 1, :3]),
    }


def metrics(targets, predictions):
    matrix = np.zeros((7, 7), dtype=int)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target, prediction] += 1
    support = matrix.sum(axis=1)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(7), where=support > 0)
    return {
        'count': int(matrix.sum()),
        'correct': int(matrix.trace()),
        'accuracy': float(matrix.trace() / matrix.sum()),
        'balanced_accuracy_present_classes': float(recalls[support > 0].mean()),
        'per_class': {
            name: {'count': int(support[i]), 'accuracy': float(recalls[i])}
            for i, name in enumerate(OPTIONS)
        },
        'confusion_matrix_target_rows': matrix.tolist(),
    }


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_id(value):
    normalized = '/'.join(p for p in value.replace('\\', '/').casefold().split('/') if p)
    return hashlib.sha256(normalized.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.data_root / 'download-receipt.json'
    if digest(receipt_path) != RECEIPT_SHA or digest(args.train) != TRAIN_SHA:
        raise ValueError('pinned original receipt/train hash mismatch')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in args.train.read_text(encoding='utf-8').splitlines()]
    if (len(rows) != 275 or any(r['split'] != 'train' for r in rows)
            or len({r['id'] for r in rows}) != len(rows)
            or len({r['group_id'] for r in rows}) != len(rows)
            or any(tuple(r['options']) != OPTIONS for r in rows)):
        raise ValueError('unexpected frozen train manifest')
    by_id = {r['id']: r for r in rows}
    source = subprocess.check_output([
        'git', 'show', f'{SOURCE_COMMIT}:scripts/audit_droid100_groups.py',
    ])
    namespace = {'__name__': 'pinned_droid_parser'}
    exec(compile(source, 'pinned_droid_parser', 'exec'), namespace)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'pinned-parser.py').write_bytes(source)
    protocol = {
        'started_utc': datetime.now(UTC).isoformat(),
        'parser_commit': SOURCE_COMMIT,
        'parser_sha256': hashlib.sha256(source).hexdigest(),
        'script_sha256': digest(__file__),
        'train_sha256': TRAIN_SHA, 'receipt_sha256': RECEIPT_SHA,
        'rules': ['train_majority', 'previous_command',
                  'sparse_displacement', 'dense_displacement'],
        'threshold': 1e-4,
        'threshold_units': 'native command units or native displacement units, not equated',
        'scope': '275 frozen train episodes only; no fitting or threshold search',
        'extra_inputs': ['previous_command', 'dense_displacement'],
        'validation_policy': 'nonselected opaque records skipped after source identity parsing',
        'probability_metrics': 'not defined for these deterministic rules',
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.perf_counter()
    predictions, shard_receipts, seen = [], [], set()
    skipped = 0
    for item in receipt['objects']:
        if '.tfrecord-' not in item['name']:
            continue
        path = args.data_root / item['name']
        if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError(f'original shard integrity failure: {path.name}')
        total = selected = 0
        for record in namespace['iter_tfrecord'](path):
            total += 1
            context = namespace['episode_example_context'](record)
            row_id = canonical_id(context['episode_metadata/file_path'])
            if row_id not in by_id:
                skipped += 1
                continue
            if row_id in seen:
                raise ValueError('duplicate selected source episode')
            row = by_id[row_id]
            if canonical_id(context['episode_metadata/recording_folderpath']) != row['group_id']:
                raise ValueError('recording identity mismatch')
            positions = np.asarray(namespace['episode_example_float_values'](
                record, 'steps/observation/cartesian_position')).reshape(-1, 6)
            actions = np.asarray(namespace['episode_example_float_values'](
                record, 'steps/action_dict/cartesian_velocity')).reshape(-1, 6)
            if len(actions) != row['source_step_count'] or len(positions) != len(actions):
                raise ValueError('source step count mismatch')
            target = direction(actions[row['target_index'], :3])
            if OPTIONS[target] != row['target_option_id']:
                raise ValueError('frozen target recomputation mismatch')
            prediction = temporal_predictions(
                positions, actions, row['input_indices'], row['target_index'])
            predictions.append({
                'id': row_id, 'group_id': row['group_id'], 'target': target,
                'predictions': prediction,
                'sparse_gap_steps': row['input_indices'][-1] - row['input_indices'][-2],
            })
            seen.add(row_id)
            selected += 1
        shard_receipts.append({
            'name': path.name, 'sha256': item['sha256'], 'bytes': item['size'],
            'records': total, 'selected_train_records': selected,
        })
        print(json.dumps(shard_receipts[-1]), flush=True)
    if seen != set(by_id):
        raise ValueError('missing selected train episodes')
    predictions.sort(key=lambda r: r['id'])
    targets = [r['target'] for r in predictions]
    majority = Counter(targets).most_common(1)[0][0]
    results = {
        'train_majority': metrics(targets, [majority] * len(targets)),
        **{rule: metrics(targets, [r['predictions'][rule] for r in predictions])
           for rule in protocol['rules'][1:]},
    }
    prediction_path = args.output / 'train-predictions.jsonl'
    prediction_path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in predictions))
    if digest(args.train) != TRAIN_SHA or digest(receipt_path) != RECEIPT_SHA:
        raise ValueError('metadata changed during read-only audit')
    result = {
        **protocol, 'completed_utc': datetime.now(UTC).isoformat(),
        'wall_seconds': time.perf_counter() - started,
        'selected_count': len(seen), 'nonselected_records_skipped': skipped,
        'target_mismatches': 0, 'shards': shard_receipts, 'metrics': results,
        'sparse_gap_steps': dict(sorted(
            Counter(r['sparse_gap_steps'] for r in predictions).items())),
        'predictions_sha256': digest(prediction_path),
        'unchanged_metadata_hashes': True, 'gpu_or_models_used': False,
    }
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['accuracy'] for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
