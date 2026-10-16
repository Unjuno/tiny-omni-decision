"""Execute full TRAIN metadata plan and fitted initialization; no optimizer or quality test."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import psutil

from scripts.greatest_hits_matched import MODALITIES, build_plan, digest, presentation
from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS, learning_rate
from scripts.native_video_cache import sha
from scripts.probe_greatest_hits_transport import AUDIO, REPO, ROOT, TASKS, VIDEO, read, write

COMMON = ROOT/'native-common-corpus-02'
FIXTURES = ROOT/'greatest-hits-transport-smoke-01'
CONTROL = ROOT/'native-long-budget-fit-01'


def normalized(row, joint):
    if joint:
        return dict(source='greatest-hits', modality='audio_video', task=row['task'],
            observation=row['event_id'], asset=row['asset_sha256'], group=row['date_group'],
            label=row['options'][row['target']])
    asset = (row['observation'].rsplit(':', 1)[0]
             if row['source'] == 'starss' else row['observation'])
    return dict(source=row['source'], modality=row['modality'], task=row['task'],
        observation=row['observation'], asset=None if row['modality'] == 'text' else asset,
        group=row['group'], label=row['options'][row['target_index']])


def accounting(ids, metadata):
    rows = [metadata[i] for i in ids]
    return dict(consumed=len(ids), unique_questions=len(set(ids)),
        repeated_questions=len(ids)-len(set(ids)),
        unique_observations=len({(r['source'], r['observation']) for r in rows}),
        unique_media_assets=len({(r['source'], r['asset'])
                                for r in rows if r['asset'] is not None}),
        unique_groups=len({(r['source'], r['group']) for r in rows}),
        modalities=dict(sorted(Counter(r['modality'] for r in rows).items())),
        sources=dict(sorted(Counter(r['source'] for r in rows).items())),
        tasks=dict(sorted(Counter(r['source']+'|'+r['task'] for r in rows).items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for metadata and CPU initialization preflight')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('clean committed source required')
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    pins, artifacts = {}, {}
    result = dict(status='RUNNING', command=[sys.executable, *sys.argv], pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                               text=True).strip(), pins=pins, artifacts=artifacts,
        optimizer_updates=0, encoder_updates=0, validation_metadata_loaded=True,
        validation_examples_forwarded=0, quality_evaluated=False, calibration_loaded=False,
        sealed_audit_loaded=False, cloud_cost=0, full_training_admitted=False)

    def pin(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('changed pinned input: '+str(path))
        pins[str(path)] = actual

    def save(name, value):
        write(args.output/name, value)
        artifacts[name] = sha(args.output/name)

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        if (time.perf_counter()-started > 900 or result['peak_working_set_bytes'] > 8*2**30
                or shutil.disk_usage(args.output).free < 16*2**30):
            raise RuntimeError('preflight resource guard')

    try:
        guard()
        pin(COMMON/'result.json',
            '47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49')
        pin(TASKS/'result.json', '7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831')
        pin(FIXTURES/'result.json',
            '7f6b60c0d4dfa40cb9a86096301f6e5757a0e57ab6b3975b19d74ae4db8167d3')
        pin(CONTROL/'best.pt', 'd7b666cd7ea60dcf236bd06e53b0107e0d5f74193824c1f4b44bbe4d5cdc1b56')
        pin(CONTROL/'result.json')
        control = read(CONTROL/'result.json')
        if (control['status'] != 'PASS_NATIVE_LONG_BUDGET_TRAIN' or control['best_step'] != 8192
                or control['step'] != 32768 or control['artifacts']['best.pt']
                != pins[str(CONTROL/'best.pt')]):
            raise ValueError('unchanged complete selected control required')
        common_receipt, tasks, fixtures = [read(p/'result.json') for p in (COMMON, TASKS, FIXTURES)]
        if (common_receipt['status'] != 'PASS_COMPLETE_NATIVE_COMMON_CORPUS'
                or tasks['status'] != 'PASS_FIXED_DEVELOPMENT_TASKS'
                or fixtures['status'] != 'PASS_REAL_FEATURE_TRANSPORT'):
            raise ValueError('passing pinned corpus and fixtures required')
        for name in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'],
                                            cwd=REPO, text=True).split('\0'):
            if name:
                pin(REPO/name)
        protocol = (REPO/'docs/pretrained_reuse'
                    /'R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md')
        pin(protocol)
        data = {}
        for kind, folder, receipt in (('joint', TASKS, tasks), ('common', COMMON, common_receipt)):
            for split in ('train', 'validation'):
                path = folder/(split+'.jsonl')
                pin(path, receipt['artifacts'][path.name])
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                if (any(r['split'] != split for r in rows)
                        or len({r['id'] for r in rows}) != len(rows)):
                    raise ValueError('unique fixed split identities required')
                data[kind, split] = rows
        if ([len(data[k, s]) for k in ('joint', 'common') for s in ('train', 'validation')]
                != [23707, 8875, 100580, 24346]):
            raise ValueError('complete prospective corpora required')
        train_ids = {r['id'] for k in ('joint', 'common') for r in data[k, 'train']}
        val_ids = {r['id'] for k in ('joint', 'common') for r in data[k, 'validation']}
        if train_ids & val_ids:
            raise ValueError('cross-split question identity overlap')
        joint, common = data['joint', 'train'], data['common', 'train']
        plan = build_plan(joint, common)
        reverse = build_plan(joint[::-1], common[::-1])
        if plan != reverse:
            raise ValueError('input order changed actual sampling')
        del reverse
        save('sampling.json', plan)
        all_rows = {r['id']: r for r in [*joint, *common]}
        metadata = {r['id']: normalized(r, True) for r in joint}
        metadata.update({r['id']: normalized(r, False) for r in common})
        flat = [i for b in plan['order'] for i in b]
        save('coverage.json', {str(step): accounting(flat[:8*step], metadata)
                              for step in EVALUATION_STEPS})
        classes, used_classes = defaultdict(set), Counter()
        consumed_unique = defaultdict(set)
        for ident, row in metadata.items():
            classes[row['source'], row['task'], row['label']].add(ident)
        permutation_hashes = []
        for occurrence, ident in enumerate(flat):
            r = metadata[ident]
            key = r['source'], r['task'], r['label']
            used_classes[key] += 1
            consumed_unique[key].add(ident)
            p = presentation(all_rows[ident], occurrence)
            permutation_hashes.append(digest([ident, p]))
        save('class-support.json', [dict(source=k[0], task=k[1], label=k[2], available=len(v),
            consumed=used_classes[k], consumed_unique=len(consumed_unique[k]))
            for k, v in sorted(classes.items())])
        save('lr-plan.json', [learning_rate(step) for step in range(1, MAX_STEPS+1)])
        save('validation-ids.json', {k: [r['id'] for r in data[k, 'validation']]
                                     for k in ('joint', 'common')})
        guard()
        import torch

        from scripts.greatest_hits_initialization import ARMS, initialized_model
        from scripts.greatest_hits_transport import bind_caches
        from scripts.greatest_hits_transport_store import ExplicitEventStore

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        checkpoint = torch.load(CONTROL/'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['step'] != 8192:
            raise ValueError('selected checkpoint step differs from receipt')
        for folder, expected in (
            (AUDIO, '95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea'),
            (VIDEO, 'd0ee01083f70a98be68b3a71c0a9abd68ac9b4b82759cc089e0232ff32bd38db')):
            pin(folder/'result.json', expected)
        audio, video = read(AUDIO/'result.json'), read(VIDEO/'result.json')
        for folder, receipt in ((AUDIO, audio), (VIDEO, video)):
            for name, value in receipt['artifacts'].items():
                pin(folder/name, value)
        pin(VIDEO/'plan.json', video['plan_sha256'])
        binding = bind_caches(dict(train=joint, validation=data['joint', 'validation']),
            audio, video, read(AUDIO/'plan.json'), read(VIDEO/'plan.json'), phase='smoke')
        store = ExplicitEventStore(binding, dict(audio=AUDIO, video=VIDEO),
            dict(audio=audio['artifacts'], video=video['artifacts']), device='cpu')
        initial_scores, parameter_counts = {}, {}
        for arm in ARMS:
            pin(FIXTURES/(arm+'.pt'), fixtures['artifacts'][arm+'.pt'])
            state = torch.load(FIXTURES/(arm+'.pt'), map_location='cpu', weights_only=True)
            model = initialized_model(arm, checkpoint['model'], state).eval()
            parameter_counts[arm] = 195365529+sum(p.numel() for p in model.parameters())
            torch.save(model.state_dict(), args.output/(arm+'-initial.pt'))
            artifacts[arm+'-initial.pt'] = sha(args.output/(arm+'-initial.pt'))
            scores = []
            with torch.inference_mode():
                for ident in binding['questions']:
                    for mode in ('joint', 'audio', 'video'):
                        blocks, q = store.get(ident, mode=mode)
                        scores.append(dict(id=ident, mode=mode, logits=model(blocks, q).tolist()))
            if arm != 'C0' and scores != initial_scores['C0']:
                raise ValueError('fitted initial output parity differs')
            initial_scores[arm] = scores
            guard()
        save('initial-predictions.json', initial_scores)
        store.verify_files()
        for path, expected in pins.items():
            if sha(Path(path)) != expected:
                raise ValueError('source/input changed during preflight')
        result.update(status='PASS_MATCHED_AV_PREFLIGHT', updates_planned=MAX_STEPS,
            presentations_planned=len(flat), order_sha256=plan['order_sha256'],
            permutation_sequence_sha256=digest(permutation_hashes),
            initial_predictions_verified=sum(map(len, initial_scores.values())),
            parameter_counts=parameter_counts, initialization='control step8192; fresh optimizer',
            never_consumed_classes=sum(used_classes[k] == 0 for k in classes),
            common_modalities=list(MODALITIES),
            existing_control_optimizer_loaded_for_training=False)
        guard()
    except BaseException as exc:
        result.update(status='INCOMPLETE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(UTC).isoformat())
        write(args.output/'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
