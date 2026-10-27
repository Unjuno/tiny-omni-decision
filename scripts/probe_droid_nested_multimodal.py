"""Frozen pretrained text probe on the fixed DROID training cohort."""

import argparse
import json
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
PROTOCOL = Path('docs/pretrained_reuse/R2_DROID_NESTED_MULTIMODAL_PROTOCOL_20261028.md')
AUDIT_SHA = 'dea0dfa4a0eefd24afaad5278f2a23662626b7573ae03e7bd9a66d55397b27f8'
REPO = Path(__file__).resolve().parents[1]
MODEL = Path('C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model')
RECEIPT = MODEL.parent / 'result-retry.json'


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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    receipt = json.loads(RECEIPT.read_bytes())
    if receipt['revision'] != '233902d25c440f23af6f7d6e94d2946bac0bee0a':
        raise ValueError('unexpected model revision')
    if any(digest(MODEL / name) != entry['sha256'] for name, entry in receipt['files'].items()):
        raise ValueError('model or tokenizer hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    values = np.zeros((len(texts), 385), dtype=np.float32)
    indices = [i for i, text in enumerate(texts) if text.strip()]
    lengths = [len(tokenizer.encode(texts[i], add_special_tokens=True)) for i in indices]
    for start in range(0, len(indices), 16):
        selected = indices[start:start + 16]
        tokens = tokenizer([texts[i] for i in selected], padding=True, truncation=True,
                           max_length=128, return_tensors='pt')
        with torch.no_grad():
            output = model(**tokens, output_hidden_states=True)
            values[selected, :384] = output.hidden_states[-1][:, 0, :].numpy()
    values[:, 384] = [not bool(text.strip()) for text in texts]
    print(f'texts exceeding 128 tokens: {sum(length > 128 for length in lengths)}', flush=True)
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


def fit_predict(sensor, video, text, y, fit, held, arm, steps):
    parts = []
    for name, data in (('sensor', sensor), ('video', video)):
        if name == 'video' and 'video' not in arm:
            continue
        mean, std = data[fit].mean(axis=0), data[fit].std(axis=0)
        std[std < 1e-6] = 1
        parts.append(np.clip((data - mean) / std, -10, 10))
    if 'text' in arm:
        parts.append(text)
    x = np.concatenate(parts, axis=1).astype(np.float32)
    torch.manual_seed(17)
    model = torch.nn.Linear(x.shape[1], 7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    X, Y = torch.tensor(x[fit]), torch.tensor(y[fit], dtype=torch.long)
    predictions = {}
    for step in range(1, max(steps) + 1):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), Y)
        loss.backward()
        optimizer.step()
        if step in steps:
            with torch.no_grad():
                predictions[step] = torch.softmax(model(torch.tensor(x[held])), dim=1).numpy()
    return predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError('new external output required')
    pinned = {TRAIN: TRAIN_SHA, AUDIT: AUDIT_SHA, PARSER: PARSER_SHA,
              RECEIPT: 'b3ff45ab20ffae8c19bcbe02d206bc8546e5687599aa9635c42c84f5fddf8d6a',
              DATA / 'download-receipt.json': RECEIPT_SHA,
              CACHE / 'sensor_sequences.npy': SENSOR_SHA, CACHE / 'labels.npy': LABEL_SHA,
              CACHE / 'video_features.npy': (
                  'a445bd638ca3500b64360215f1033f63e381e2aa38df9856b38dd6b912041585')}
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
    raw_video = np.array(np.load(CACHE / 'video_features.npy', mmap_mode='r')[:275])
    if raw_video.shape != (275, 3, 8, 768):
        raise ValueError('video shape mismatch')
    views = raw_video.mean(axis=1)
    video = np.concatenate((views[:, :4].mean(axis=1), views[:, 4:].mean(axis=1)), axis=1)
    arms = ('sensor', 'sensor_video', 'sensor_text', 'sensor_video_text')
    pred = {arm: np.zeros((275, 7), dtype=np.float32) for arm in arms}
    present = np.array([bool(t.strip()) for t in texts])
    torch.set_num_threads(1)
    selections = []
    steps = (1, 10, 50, 100, 200)
    for f in range(5):
        train, held = fold != f, fold == f
        if set(groups[train]) & set(groups[held]) or len(np.unique(y[train])) != 7:
            raise ValueError('fold integrity or class coverage failure')
        inner_groups = np.unique(groups[train])
        np.random.default_rng(17).shuffle(inner_groups)
        inner_map = {int(g): i % 2 for i, g in enumerate(inner_groups)}
        inner = np.array([inner_map.get(int(g), -1) for g in groups])
        record = {'outer_fold': f, 'arms': {}}
        for arm in arms:
            pooled = {step: np.zeros((275, 7), dtype=np.float32) for step in steps}
            for k in range(2):
                inner_fit, inner_held = train & (inner != k), train & (inner == k)
                if len(np.unique(y[inner_fit])) != 7:
                    raise ValueError('inner training fold lacks a class')
                estimates = fit_predict(sensor, video, text_x, y, inner_fit, inner_held, arm, steps)
                for step in steps:
                    pooled[step][inner_held] = estimates[step]
            scores = {step: metrics(pooled[step][train], y[train])['nll'] for step in steps}
            best = min(steps, key=lambda s: (scores[s], s))
            pred[arm][held] = fit_predict(sensor, video, text_x, y, train, held, arm, (best,))[best]
            record['arms'][arm] = {'inner_nll': scores, 'selected_update': best}
        selections.append(record)
        print(f'completed fold {f}', flush=True)
    if any(digest(p) != h for p, h in pinned.items()):
        raise ValueError('input changed during run')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'oof.npz', labels=y, fold=fold, groups=groups,
                        present=present, **pred)
    report = {'selections': selections, 'counts': {'train': len(rows),
                         'text_present': int(present.sum()),
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
