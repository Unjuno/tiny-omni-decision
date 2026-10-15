"""Freeze a state-disjoint semantic corpus and full existing retention validation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.condition_binding_corpus import identity, partition, questions_for, schedule
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.semantic_state_diagnostic import oracle_state

CONFIG = Path("configs/pretrained_reuse/condition_binding_v1.json")


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import yaml

    from scripts.train_frozen_text_probe import _load_examples
    from scripts.train_numeric_deterministic import verify_splits
    from tiny_omni_decision.schema import DatasetManifest

    config = read(CONFIG)
    root = Path(config['root'])
    pins = {str(root / p): h for p, h in config['source_pins'].items()}
    pins['manifests/dataset.example.yaml'] = (
        '60c29a1efc4d93179422cf5acc7c2f38949b603efce5dc7a5073a36cd3e3ccdd')
    verify(pins)
    annotation = lines(root / 'clevr4-full-coverage-seed17/train-labels.jsonl')
    if any(r['split'] != 'train' for r in annotation):
        raise ValueError('non-TRAIN source')
    previous = read(root / 'semantic-state-bridge-01/states.json')
    states = partition([oracle_state(r) for r in annotation], sum(previous.values(), []))
    datasets = {name: [r for state in states[name] for r in questions_for(state, name)]
                for name in ('train', 'validation')}
    for q in (False, True):
        for o in (False, True):
            datasets[f'transfer-q{int(q)}-o{int(o)}'] = [
                r for state in states['transfer'] for r in questions_for(state, 'transfer', q, o)]
    manifest = DatasetManifest.model_validate(yaml.safe_load(
        Path('manifests/dataset.example.yaml').read_text(encoding='utf-8')))

    def original(split):
        path = root / f'text/{split}.jsonl'
        kinds = {r['state_id'] + ':' + k: definition['type'] for r in lines(path)
                 for k, definition in json.loads(r['questions']).items()}
        return [dict(id=e.id, state=e.state, question=e.question, options=e.options,
                     target=e.target, group=e.source_record_id.split(':')[0],
                     task=kinds[e.source_record_id], source='original', split=split)
                for e in _load_examples(path, manifest.model_copy(update={'split':split}), True)]

    old_train, old_val = original('train'), original('validation')
    old_curriculum = [dict(r, source='curriculum') for r in lines(
        root / 'conditional-curriculum-01/validation.jsonl')]
    parent = read(root / 'text-coverage-training-01/config.json')
    if [[r['id'] for r in v] for v in (old_val, old_curriculum)] != parent['validation_ids']:
        raise ValueError('retention validation IDs/order changed')
    replay = sorted(old_train, key=lambda r: hashlib.sha256(
        f"17:binding-replay:{r['id']}".encode()).hexdigest())[:2048]
    datasets.update(replay=replay, original=old_val, curriculum=old_curriculum)
    verify_splits(datasets['train'] + replay,
                  datasets['validation'] + old_val + old_curriculum + datasets['transfer-q0-o0'])
    normalized = {name: {r['state'] for r in rows} for name, rows in datasets.items()}
    if any(normalized['train'] & normalized[name] for name in (
            'validation', 'transfer-q0-o0')):
        raise ValueError('fact-state overlap')
    known = {identity(r) for r in annotation}
    if any(identity(s) in known for name in ('validation','transfer') for s in states[name]):
        raise ValueError('unused-state constraint failed')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in datasets.items():
        (args.output / f'{name}.jsonl').write_text(
            ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows), encoding='utf-8')
    write(args.output / 'sample-order.json', schedule(len(datasets['train']), len(replay)))
    write(args.output / 'state-partitions.json', states)
    verify(pins)
    write(args.output / 'result.json', dict(
        status='PASS_FROZEN_DATA_CONTRACT', source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'], text=True).strip(), inputs=pins,
        code={p: sha(p) for p in (str(CONFIG), __file__, 'scripts/condition_binding_corpus.py')},
        rows={name:len(rows) for name, rows in datasets.items()},
        unique_states={name:len(values) for name, values in normalized.items()},
        known_upstream_states=len(known), unused_selector_and_transfer_states=512,
        validation_ids_match_parent=True, disjoint=True,
        hashes={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps({name:len(rows) for name, rows in datasets.items()}), flush=True)


if __name__ == '__main__':
    main()
