"""Pinned frozen language features: train-only smoke, full cache, or full fresh replay."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.native_language_cache import inventory, storage_plan, validate_profile
from scripts.native_training_readiness import TEXT
from scripts.native_video_cache import sha
from tiny_omni_decision.native_language import candidate_vector, observation_tokens

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs/pretrained_reuse/native_language_cache_v1.json'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def verify(pins):
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError('source/input hash changed: ' + path)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['smoke', 'cache', 'replay'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if ((args.phase != 'smoke') != bool(args.reference)
            or bool(args.reference) != bool(args.reference_sha256)):
        raise ValueError('cache requires smoke receipt; replay requires cache receipt and SHA')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('clean committed source required')
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(status='RUNNING', phase=args.phase, started_utc=datetime.now(UTC).isoformat(),
                  command=sys.argv, source_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  config_sha256=sha(CONFIG), optimizer_updates=0, quality_evaluated=False,
                  calibration_loaded=False, sealed_audit_loaded=False, cloud_cost=0,
                  completed_states=0, completed_queries=0)
    write(args.output / 'result.json', result)
    banks = {}
    try:
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        validate_profile(config)
        pins = config['input_pins'].copy()
        names = subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'], cwd=REPO,
                                        text=True).split('\0')
        pins.update({str(REPO / p): sha(REPO / p) for p in names if p})
        pins[str(CONFIG)] = sha(CONFIG)
        verify(pins)
        write(args.output / 'effective-config.json', config)
        reference = None
        if args.reference:
            if sha(args.reference / 'result.json') != args.reference_sha256:
                raise ValueError('reference receipt hash changed')
            reference = json.loads((args.reference / 'result.json').read_text(encoding='utf-8'))
            expected = 'PASS_NATIVE_LANGUAGE_SMOKE' if args.phase == 'cache' else (
                'PASS_NATIVE_LANGUAGE_CACHE')
            if (reference['status'] != expected or reference['validated_hashes'] != pins
                    or reference['config_sha256'] != result['config_sha256']):
                raise ValueError('complete reference with same inputs/source required')
            verify({str(args.reference / name): value
                    for name, value in reference['artifact_hashes'].items()})
            result['reference'] = dict(path=str(args.reference), sha256=args.reference_sha256)
        files = {name: [json.loads(line) for line in Path(config['files'][name]).read_text(
            encoding='utf-8').splitlines()] for names in TEXT.values() for name in names}
        index = inventory(files)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config['seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('registered CUDA profile unavailable')
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
        plan = storage_plan(index, tokenizer)
        write(args.output / 'index.json', plan)
        needed = plan['tensor_bytes'] if args.phase == 'cache' else 0
        if shutil.disk_usage(args.output).free < needed + config['disk_reserve_bytes']:
            raise RuntimeError('insufficient disk for full arrays plus reserve')
        language, loading = AutoModelForSequenceClassification.from_pretrained(
            config['model_path'], local_files_only=True, dtype=torch.float32,
            attn_implementation='eager', output_loading_info=True)
        if any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys']):
            raise ValueError('nonexact pinned model load')
        language.eval().requires_grad_(False).to('cuda')
        before = state_digest(language)
        result.update(validated_hashes=pins,
                      parameters=sum(p.numel() for p in language.parameters()),
                      initial_model_state_sha256=before, device=torch.cuda.get_device_name(),
                      versions={p: importlib.metadata.version(p) for p in (
                          'torch', 'transformers', 'numpy', 'safetensors', 'tokenizers')},
                      python=sys.version, cuda=torch.version.cuda, dtype='float32',
                      attention='eager', batch_size=1, tf32=False,
                      unique_states=len(plan['states']),
                      unique_queries=len(plan['queries']), tensor_bytes=plan['tensor_bytes'],
                      source_rows={k: len(v) for k, v in plan['rows'].items()},
                      long_states=sum(v['tokens'] > 512 for v in plan['states'].values()))
        result['runtime_implementations'] = {
            inspect.getfile(type(obj)): sha(Path(inspect.getfile(type(obj))))
            for obj in (language.deberta, tokenizer)}
        if reference and any(reference[key] != result[key] for key in (
                'versions', 'runtime_implementations', 'initial_model_state_sha256')):
            raise ValueError('reference runtime/model state differs')
        if result['parameters'] != config['parameters']:
            raise ValueError('pinned parameter count differs')
        if args.phase == 'cache':
            banks = {name: np.lib.format.open_memmap(
                args.output / name, mode='w+', dtype=np.float32, shape=tuple(shape))
                for name, shape in plan['shapes'].items()}
        elif args.phase == 'replay':
            prior = json.loads((args.reference / 'index.json').read_text(encoding='utf-8'))
            if prior != plan:
                raise ValueError('reference mapping differs')
            banks = {name: np.load(args.reference / name, mmap_mode='r', allow_pickle=False)
                     for name in plan['shapes']}
        state_keys = list(plan['states'])
        pair_keys = list(plan['queries'])
        if args.phase == 'smoke':
            train = [k for k in state_keys if plan['states'][k]['split'] == 'train']
            longest = max(train, key=lambda k: plan['states'][k]['tokens'])
            state_keys = list(dict.fromkeys([*train[:3], longest]))
            pair_keys = sorted({q for name in TEXT['train'] for row in index['rows'][name]
                                for q in row['queries']})[:8]
        result['state_order_sha256'] = hashlib.sha256('\n'.join(state_keys).encode()).hexdigest()
        result['query_order_sha256'] = hashlib.sha256('\n'.join(pair_keys).encode()).hexdigest()
        result['expected_states'], result['expected_queries'] = len(state_keys), len(pair_keys)
        result['max_replay_abs'] = 0.0
        segments = {}
        phase_tick = time.perf_counter()

        def guard():
            if time.perf_counter() - tick > config['max_wall_seconds']:
                raise RuntimeError('registered wall budget exceeded')
            if shutil.disk_usage(args.output).free < config['disk_reserve_bytes']:
                raise RuntimeError('disk reserve breached')
            if torch.cuda.max_memory_allocated() > config['max_vram_bytes']:
                raise RuntimeError('registered VRAM budget exceeded')
            memory = psutil.Process().memory_info()
            result['os_peak_working_set_bytes'] = memory.peak_wset
            if memory.peak_wset > config['max_rss_bytes']:
                raise RuntimeError('registered host memory budget exceeded')

        def store(kind, key, metadata, value):
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('nonfinite/non-FP32 features')
            if kind == 'state':
                if value.shape != (metadata['tokens'], 384):
                    raise ValueError('state token coverage changed')
                selection = slice(metadata['offset'], metadata['offset'] + metadata['tokens'])
            else:
                if value.shape != (384,):
                    raise ValueError('query shape changed')
                selection = metadata['offset']
            if args.phase == 'smoke':
                path = args.output / (kind + '-' + key + '.npy')
                with path.open('xb') as handle:
                    np.save(handle, value, allow_pickle=False)
                if not np.array_equal(np.load(path, allow_pickle=False), value):
                    raise ValueError('smoke save/reload differs')
            elif args.phase == 'cache':
                banks[metadata['file']][selection] = value
            else:
                expected = banks[metadata['file']][selection]
                delta = float(np.max(np.abs(value - expected)))
                result['max_replay_abs'] = max(delta, result['max_replay_abs'])
                if not np.array_equal(value, expected):
                    raise ValueError('full fresh feature replay differs')

        for key in state_keys:
            guard()
            hidden, spans = observation_tokens(language, tokenizer, index['states'][key])
            store('state', key, plan['states'][key], hidden.cpu().numpy())
            segments[key] = spans
            result['completed_states'] += 1
            if result['completed_states'] % 128 == 0:
                write(args.output / 'result.json', result)
                print('states', result['completed_states'], '/', len(state_keys), flush=True)
        result['state_encoding_seconds'] = time.perf_counter() - phase_tick
        phase_tick = time.perf_counter()
        for key in pair_keys:
            guard()
            question, candidate = index['pairs'][key]
            value = candidate_vector(language, tokenizer, question, candidate)
            store('query', key, plan['queries'][key], value.cpu().numpy())
            result['completed_queries'] += 1
            if result['completed_queries'] % 512 == 0:
                write(args.output / 'result.json', result)
                print('queries', result['completed_queries'], '/', len(pair_keys), flush=True)
        result['query_encoding_seconds'] = time.perf_counter() - phase_tick
        if args.phase == 'smoke':
            # Compare short text against the actual previously saved native input.
            with torch.inference_mode():
                native = observation_tokens(language, tokenizer, config['native_text'])[0]
                expected = np.load(config['native_tokens'], allow_pickle=False)
                result['native_short_max_abs'] = float(
                    np.max(np.abs(native.cpu().numpy() - expected)))
                if result['native_short_max_abs'] != 0:
                    raise ValueError('historical short observation is not exact')
                question, candidate = index['pairs'][pair_keys[0]]
                encoded = tokenizer([question], text_pair=[candidate], return_tensors='pt',
                                    padding=True, truncation=False).to('cuda')
                original = language.deberta(**encoded).last_hidden_state
                original = (original * encoded['attention_mask'][..., None]).sum(1)
                original /= encoded['attention_mask'].sum(1, keepdim=True)
                adapted = candidate_vector(language, tokenizer, question, candidate)
                result['single_pair_original_max_abs'] = (original[0]-adapted).abs().max().item()
                if result['single_pair_original_max_abs'] != 0:
                    raise ValueError('original single-pair convention differs')
        if args.phase == 'cache':
            for bank in banks.values():
                bank.flush()
            banks.clear()
            for name, shape in plan['shapes'].items():
                bank = np.load(args.output / name, mmap_mode='r', allow_pickle=False)
                if bank.shape != tuple(shape) or bank.dtype != np.float32:
                    raise ValueError('saved bank format differs')
                for offset in range(0, len(bank), 1024):
                    if not np.isfinite(bank[offset:offset+1024]).all():
                        raise ValueError('saved bank is nonfinite')
        write(args.output / 'segments.json', segments)
        after = state_digest(language)
        if before != after:
            raise ValueError('frozen encoder state changed')
        result['final_model_state_sha256'] = after
        verify(pins)
        result['artifact_hashes'] = {p.name: sha(p) for p in args.output.iterdir()
                                     if p.is_file() and p.name != 'result.json'}
        guard()
        result.update(status='PASS_NATIVE_LANGUAGE_' + args.phase.upper(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      finished_utc=datetime.now(UTC).isoformat(), seconds=time.perf_counter()-tick)
        write(args.output / 'result.json', result)
        print(result['status'], result['seconds'], flush=True)
    except BaseException as error:
        result.update(status='INCOMPLETE', error=repr(error), traceback=traceback.format_exc(),
                      finished_utc=datetime.now(UTC).isoformat(), seconds=time.perf_counter()-tick)
        write(args.output / 'result.json', result)
        raise


if __name__ == '__main__':
    main()
