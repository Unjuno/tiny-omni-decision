"""Frozen train-only DROID instruction probe; see the fixed protocol."""

import argparse
import hashlib
import json
import re
import runpy
from pathlib import Path

import numpy as np
import torch

from scripts.audit_droid_instruction_timing import FIELDS
from scripts.audit_droid_train_all_transitions import DATA, PARSER, PARSER_SHA, TRAIN
from scripts.audit_droid_train_temporal import OPTIONS, RECEIPT_SHA, TRAIN_SHA, canonical_id, digest
from scripts.compare_droid_train_temporal import LABEL_SHA, SENSOR_SHA

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
AUDIT = ROOT / 'droid-fresh-next-action-prep-attempt-02/audit.json'
CACHE = ROOT / 'droid-fresh-next-action-features-attempt-01'
PROTOCOL = Path('docs/pretrained_reuse/R2_DROID_INSTRUCTION_PROBE_PROTOCOL_20261028.md')
AUDIT_SHA = 'dea0dfa4a0eefd24afaad5278f2a23662626b7573ae03e7bd9a66d55397b27f8'
REPO = Path(__file__).resolve().parents[1]


def group_components(instructions):
    parent = list(range(len(instructions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, values in enumerate(instructions):
        for text in values:
            if text in first:
                parent[find(i)] = find(first[text])
            else:
                first[text] = i
    return np.array([find(i) for i in range(len(instructions))])


def word_features(texts):
    values = np.zeros((len(texts), 257), dtype=np.float32)
    for i, text in enumerate(texts):
        words = re.findall(r'\w+', text.lower())
        for word in words:
            idx = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], 'big') % 256
            values[i, idx] += 1
        norm = np.linalg.norm(values[i, :256])
        if norm:
            values[i, :256] /= norm
        values[i, 256] = not bool(text.strip())
    return values


def metrics(p, y):
    chosen = p[np.arange(len(y)), y]
    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        mask = (confidence >= lo) & (confidence < lo + 0.1 + (1e-8 if lo > 0.89 else 0))
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - confidence[mask].mean())
    return {'count': len(y), 'accuracy': float((pred == y).mean()),
            'nll': float(-np.log(np.maximum(chosen, 1e-12)).mean()),
            'brier': float(((p - np.eye(7)[y]) ** 2).sum(axis=1).mean()), 'ece': float(ece)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    pinned = {TRAIN: TRAIN_SHA, AUDIT: AUDIT_SHA, PARSER: PARSER_SHA,
              DATA / 'download-receipt.json': RECEIPT_SHA,
              CACHE / 'sensor_sequences.npy': SENSOR_SHA, CACHE / 'labels.npy': LABEL_SHA}
    if any(digest(p) != h for p, h in pinned.items()):
        raise ValueError('pinned input mismatch')
    rows = [json.loads(line) for line in TRAIN.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 275 or any(r['split'] != 'train' for r in rows):
        raise ValueError('unexpected train cohort')
    ids = {r['id']: i for i, r in enumerate(rows)}
    if len(ids) != 275 or len({r['group_id'] for r in rows}) != 275:
        raise ValueError('duplicate train identity')
    audit = json.loads(AUDIT.read_bytes())['episodes']
    meta = {canonical_id(r['episode_metadata/file_path']): r for r in audit}
    if len(meta) != len(audit) or not set(ids) <= set(meta):
        raise ValueError('audit identity mismatch')
    groups = group_components([meta[r['id']]['normalized_instructions'] for r in rows])
    unique = np.unique(groups)
    np.random.default_rng(17).shuffle(unique)
    fold_for_group = {int(g): i % 5 for i, g in enumerate(unique)}
    fold = np.array([fold_for_group[int(g)] for g in groups])
    texts = [None] * len(rows)
    api = runpy.run_path(str(PARSER), run_name='pinned_droid_parser')
    for item in json.loads((DATA / 'download-receipt.json').read_bytes())['objects']:
        if '.tfrecord-' not in item['name']:
            continue
        path = DATA / item['name']
        if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError('source shard mismatch')
        for raw in api['iter_tfrecord'](path):
            context = api['episode_example_context'](raw)
            identity = canonical_id(context['episode_metadata/file_path'])
            if identity not in ids:
                continue
            i = ids[identity]
            group_id = canonical_id(context['episode_metadata/recording_folderpath'])
            if texts[i] is not None or group_id != rows[i]['group_id']:
                raise ValueError('duplicate or mismatched source')
            step = rows[i]['target_index'] - 1
            seqs = [api['episode_example_bytes_values'](raw, key) for key in FIELDS]
            if any(len(seq) != rows[i]['source_step_count'] for seq in seqs):
                raise ValueError('instruction sequence mismatch')
            texts[i] = ' '.join(seq[step].decode('utf-8').strip() for seq in seqs).strip()
        print(f'checked source shard; {sum(t is not None for t in texts)} / 275', flush=True)
    if any(t is None for t in texts):
        raise ValueError('missing source records')
    sensor = np.array(np.load(CACHE / 'sensor_sequences.npy', mmap_mode='r')[:275])
    sensor = sensor.reshape(275, 112)
    y = np.array(np.load(CACHE / 'labels.npy', mmap_mode='r')[:275], dtype=int)
    if not np.array_equal(y, [OPTIONS.index(r['target_option_id']) for r in rows]):
        raise ValueError('label mismatch')
    text_x = word_features(texts)
    pred = {arm: np.zeros((275, 7), dtype=np.float32) for arm in ('sensor', 'sensor_text')}
    present = np.array([bool(t.strip()) for t in texts])
    torch.set_num_threads(1)
    for f in range(5):
        train, held = fold != f, fold == f
        if set(groups[train]) & set(groups[held]) or len(np.unique(y[train])) != 7:
            raise ValueError('fold integrity or class coverage failure')
        mean = sensor[train].mean(axis=0)
        std = sensor[train].std(axis=0)
        std[std < 1e-6] = 1
        x = np.clip((sensor - mean) / std, -10, 10).astype(np.float32)
        for arm in pred:
            features = x if arm == 'sensor' else np.concatenate((x, text_x), axis=1)
            torch.manual_seed(17)
            model = torch.nn.Linear(features.shape[1], 7)
            optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
            X, Y = torch.tensor(features[train]), torch.tensor(y[train], dtype=torch.long)
            for _ in range(200):
                optim.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(X), Y)
                loss.backward()
                optim.step()
            with torch.no_grad():
                pred[arm][held] = torch.softmax(model(torch.tensor(features[held])), dim=1).numpy()
        print(f'completed fold {f}', flush=True)
    if any(digest(p) != h for p, h in pinned.items()):
        raise ValueError('input changed during run')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'oof.npz', labels=y, fold=fold, present=present, **pred)
    report = {'counts': {'train': len(rows), 'text_present': int(present.sum()),
                         'components': len(unique),
                         'class_counts': np.bincount(y, minlength=7).tolist()},
              'metrics': {arm: {'all': metrics(p, y), 'present': metrics(p[present], y[present]),
                                'folds': [metrics(p[fold == f], y[fold == f]) for f in range(5)]}
                          for arm, p in pred.items()},
              'input_hashes': {str(k): v for k, v in pinned.items()},
              'script_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PROTOCOL),
              'oof_sha256': digest(out / 'oof.npz'), 'torch_version': torch.__version__}
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
