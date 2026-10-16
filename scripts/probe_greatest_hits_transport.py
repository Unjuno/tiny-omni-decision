"""Real frozen TRAIN features through C0 and four untrained AV policies; CPU only."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import psutil

from scripts.greatest_hits_transport import bind_caches
from scripts.native_video_cache import sha

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
AUDIO = ROOT/'greatest-hits-audio-query-smoke-01'
VIDEO = ROOT/'greatest-hits-video-cache-smoke-01'
TASKS = ROOT/'greatest-hits-development-tasks-01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256):
        raise ValueError('paired reference path/hash required')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip()):
        raise ValueError('clean committed source and CPU-only process required')
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    pins, artifacts = {}, {}
    result = dict(status='RUNNING', command=[sys.executable, *sys.argv], pid=os.getpid(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                             text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(), optimizer_updates=0, device='cpu',
        dtype='float32', seed=17, cpu_threads=1, quality_evaluated=False,
        validation_task_metadata_loaded=True, validation_features_loaded=False,
        sealed_audit_loaded=False, calibration_loaded=False, cloud_cost=0,
        completed_questions=0, decisions=0, pins=pins, artifacts=artifacts,
        max_permutation_logit_error=0., reference_sha256=args.reference_sha256)

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('pin mismatch: '+str(path))
        pins[str(path)] = actual

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        if (time.perf_counter()-tick > 900 or result['peak_working_set_bytes'] > 4*2**30
                or shutil.disk_usage(args.output).free < 16*2**30):
            raise RuntimeError('fixed15min/4GiB/16GiB-reserve guard')

    try:
        guard()
        pin(AUDIO/'result.json',
            '95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea')
        pin(VIDEO/'result.json',
            'd0ee01083f70a98be68b3a71c0a9abd68ac9b4b82759cc089e0232ff32bd38db')
        pin(TASKS/'result.json',
            '7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831')
        audio, video, tasks = [read(p/'result.json') for p in (AUDIO, VIDEO, TASKS)]
        for directory, receipt in ((AUDIO, audio), (VIDEO, video)):
            for name, digest in receipt['artifacts'].items():
                pin(directory/name, digest)
        pin(VIDEO/'plan.json', video['plan_sha256'])
        splits = {}
        for split in ('train', 'validation'):
            path = TASKS/(split+'.jsonl')
            pin(path, tasks['artifacts'][path.name])
            splits[split] = [json.loads(line) for line in path.read_text().splitlines()]
        binding = bind_caches(splits, audio, video, read(AUDIO/'plan.json'),
                              read(VIDEO/'plan.json'), phase='smoke')
        if len(binding['events']) != 21 or len(binding['questions']) != 63:
            raise ValueError('fixed21 TRAIN events and63 questions required')
        for name in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'],
                                            cwd=REPO, text=True).split('\0'):
            if name:
                pin(REPO/name)
        reference = None
        if args.reference:
            pin(args.reference/'result.json', args.reference_sha256)
            reference = read(args.reference/'result.json')
            if (reference['status'] != 'PASS_REAL_FEATURE_TRANSPORT'
                    or reference['source_commit'] != result['source_commit']):
                raise ValueError('complete same-source reference required')
            for name, digest in reference['artifacts'].items():
                pin(args.reference/name, digest)
        write(args.output/'binding.json', binding)
        artifacts['binding.json'] = sha(args.output/'binding.json')
        import torch

        from scripts.greatest_hits_transport_store import ExplicitEventStore
        from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
        from tiny_omni_decision.native_candidate import NativeCandidateReadout, candidate_json

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        store = ExplicitEventStore(binding, dict(audio=AUDIO, video=VIDEO),
                                   dict(audio=audio['artifacts'], video=video['artifacts']))
        baseline = NativeCandidateReadout().eval()
        template = ConditionalAVReadout(copy.deepcopy(baseline)).eval()
        models, loaded = dict(C0=baseline), {}
        for exchange, gated in ((False,False), (False,True), (True,False), (True,True)):
            name = f'E{int(exchange)}G{int(gated)}'
            model = ConditionalAVReadout(exchange=exchange, gated=gated).eval()
            state = copy.deepcopy(template.state_dict())
            state['_extra_state'] = model.get_extra_state()
            model.load_state_dict(state, strict=True)
            models[name] = model
        result['parameters'] = {}
        before = {}

        def state_hash(model):
            h = hashlib.sha256()
            for name, value in sorted(model.state_dict().items()):
                h.update(name.encode())
                h.update(value.detach().cpu().numpy().tobytes() if isinstance(value, torch.Tensor)
                         else json.dumps(value, sort_keys=True).encode())
            return h.hexdigest()

        for name, model in models.items():
            count = sum(p.numel() for p in model.parameters())
            if count != (889345 if name == 'C0' else 1732420):
                raise ValueError('readout inventory differs')
            result['parameters'][name] = dict(readout=count, encoder_inclusive=195365529+count)
            if name != 'C0':
                result['parameters'][name].update(model.parameter_ledger())
                for key, value in template.state_dict().items():
                    if key != '_extra_state':
                        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            path = args.output/(name+'.pt')
            torch.save(model.state_dict(), path)
            artifacts[path.name] = sha(path)
            restored = NativeCandidateReadout() if name == 'C0' else ConditionalAVReadout(
                exchange=model.get_extra_state()['exchange'],
                gated=model.get_extra_state()['gated'])
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True),
                                      strict=True)
            loaded[name] = restored.eval()
            before[name] = state_hash(model)
            if state_hash(restored) != before[name]:
                raise ValueError('reloaded model state differs')
        result.update(frozen_readout_before=before,
            versions={p: importlib.metadata.version(p) for p in ('torch','numpy')},
            python=platform.python_version(), load_seconds=time.perf_counter()-tick)
        write(args.output/'result.json', result)
        predict_tick = time.perf_counter()
        with (args.output/'predictions.jsonl').open('x', encoding='utf-8') as stream, \
                torch.inference_mode():
            for ident, question in binding['questions'].items():
                guard()
                options = question['options']
                order = list(reversed(range(len(options))))
                for mode in ('joint','audio','video'):
                    blocks, q = store.get(ident, mode=mode)
                    swapped_blocks, swapped_q = store.get(ident, mode=mode, order=order)
                    expected = baseline(blocks, q)
                    for name, model in models.items():
                        if name == 'C0':
                            scores, gates = expected, None
                        else:
                            scores, diagnostic = model(blocks, q, diagnostics=True)
                            gates = diagnostic['gates']
                            if diagnostic['active'] != (mode == 'joint'):
                                raise ValueError('missing-input bypass differs')
                        torch.testing.assert_close(scores, expected, rtol=0, atol=0)
                        reloaded = loaded[name](blocks,q)
                        torch.testing.assert_close(reloaded, scores, rtol=0, atol=0)
                        swapped = model(swapped_blocks, swapped_q)
                        difference = float((swapped-scores[order]).abs().max())
                        result['max_permutation_logit_error'] = max(
                            result['max_permutation_logit_error'], difference)
                        torch.testing.assert_close(swapped, scores[order], rtol=0, atol=1e-6)
                        decision = candidate_json(scores, options)
                        permutation = candidate_json(swapped, [options[i] for i in order])
                        if (abs(sum(decision['probabilities'].values())-1) > 1e-12
                                or decision['selected_option_id']
                                != permutation['selected_option_id']
                                or max(abs(v-permutation['probabilities'][key])
                                       for key,v in decision['probabilities'].items()) > 1e-6):
                            raise ValueError('candidate-associated probabilities differ')
                        row = dict(id=ident, event_id=question['event_id'], arm=name, mode=mode,
                            options=options, logits=scores.tolist(), decision=decision,
                            permuted_logits=swapped.tolist(), permutation=order,
                            permuted_decision=permutation,
                            gates=gates.tolist() if gates is not None else None)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        result['decisions'] += 1
                result['completed_questions'] += 1
                if result['completed_questions'] % 3 == 0:
                    result['elapsed_seconds'] = time.perf_counter()-tick
                    write(args.output/'result.json', result)
                    print(result['completed_questions'], '/ 63', ident, flush=True)
        result['prediction_seconds'] = time.perf_counter()-predict_tick
        artifacts['predictions.jsonl'] = sha(args.output/'predictions.jsonl')
        result['frozen_readout_after'] = {name: state_hash(m) for name,m in models.items()}
        if (result['frozen_readout_after'] != before or result['completed_questions'] != 63
                or result['decisions'] != 945):
            raise ValueError('incomplete coverage or readout state changed')
        store.verify_files()
        for path, digest in pins.items():
            if sha(Path(path)) != digest:
                raise ValueError('source/input changed during probe')
        if reference and (reference['artifacts'] != artifacts
                or reference['frozen_readout_before'] != before
                or reference['versions'] != result['versions']):
            raise ValueError('fresh process replay differs')
        guard()
        result['status'] = 'PASS_REAL_FEATURE_TRANSPORT'
    except BaseException as exc:
        result.update(status='INCOMPLETE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-tick, finished_utc=datetime.now(UTC).isoformat())
        write(args.output/'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
