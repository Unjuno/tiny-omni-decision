"""Frozen native audio/query cache with independent smoke/replay gates; no learning."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import psutil

from scripts.greatest_hits_audio_query import (
    audio_positions,
    expected_reference_matches,
    query_inventory,
    store_audio_event,
)
from scripts.greatest_hits_cache import frozen_events
from scripts.greatest_hits_features import prepared_audio_tokens
from scripts.greatest_hits_inputs import crop_audio, resample_direct
from scripts.native_video_cache import sha
from scripts.probe_greatest_hits_encoders import check_source

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
TASKS = ROOT/'greatest-hits-development-tasks-01'
MEDIA = ROOT/'greatest-hits-development-media-01/members'
REVIEW = ROOT/'greatest-hits-source-review-01'
INPUT = ROOT/'greatest-hits-input-physical-01'
AUDIO = ROOT/'greatest-hits-encoder-audio-01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('reference', 'smoke', 'smoke-replay'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    for name in ('reference', 'smoke', 'smoke_replay'):
        if bool(getattr(args, name)) != bool(getattr(args, name+'_sha256')):
            raise ValueError('reference paths require receipt SHA-256')
    if (args.phase == 'full') != bool(args.smoke and args.smoke_replay):
        raise ValueError('full cache requires successful smoke and fresh replay')
    if args.phase == 'smoke' and (args.smoke or args.smoke_replay):
        raise ValueError('smoke cannot have full-mode gate inputs')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('clean committed source required')
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    pins, artifacts = {}, {}
    result = dict(status='RUNNING', phase=args.phase, pid=os.getpid(),
        command=[sys.executable, *sys.argv], source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(), completed_events=0, completed_parents=0,
        completed_questions=0, matched_review_events=0, matched_old_audio_events=0,
        matched_smoke_events=0, optimizer_updates=0, dtype='float32', attention='eager',
        audio_forward_seconds=0., audio_decode_seconds=0., query_forward_seconds=0.,
        validation_task_metadata_loaded=True, validation_media_loaded=args.phase == 'full',
        sealed_audit_loaded=False, calibration_loaded=False, quality_evaluated=False,
        cloud_cost=0, artifacts=artifacts, parents=[], pins=pins,
        reference_sha256=args.reference_sha256, smoke_sha256=args.smoke_sha256,
        smoke_replay_sha256=args.smoke_replay_sha256)
    torch = None

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('pin mismatch: '+str(path))
        pins[str(path)] = actual

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        result['peak_vram_bytes'] = torch.cuda.max_memory_allocated() if torch else 0
        if (time.perf_counter()-tick > (900 if args.phase == 'smoke' else 7200)
                or result['peak_working_set_bytes'] > 6*2**30
                or result['peak_vram_bytes'] > 3*2**30
                or shutil.disk_usage(args.output).free < 16*2**30):
            raise RuntimeError('fixed wall/host/GPU/disk resource guard')

    def save(name, value):
        value = np.asarray(value)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('finite FP32 cache tensors required')
        path = args.output/(name+'.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        np.testing.assert_array_equal(value, np.load(path, allow_pickle=False))
        artifacts[path.name] = sha(path)
        if reference and (artifacts[path.name] != reference['artifacts'].get(path.name)
                          or sha(args.reference/path.name) != artifacts[path.name]):
            raise ValueError('fresh-process tensor replay differs: '+path.name)

    def receipt(path, expected):
        pin(path/'result.json', expected)
        value = read(path/'result.json')
        if (value['status'] != 'PASS_COMPLETE_AUDIO_QUERY_CACHE'
                or value['source_commit'] != result['source_commit']):
            raise ValueError('complete same-source cache receipt required')
        for name, digest in value['artifacts'].items():
            pin(path/name, digest)
        return value

    try:
        guard()
        pin(TASKS/'result.json',
            '7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831')
        pin(REVIEW/'result.json',
            '7f68821211c364cb84eaa24d82e41ca3a5a25d9cafb6a30e9e365dcbdfcee2be')
        pin(INPUT/'result.json',
            '97c91689effda1e7a080b81ddd4be7738718168493137a565f821371d493c0f6')
        pin(AUDIO/'result.json',
            'b0035843449a66c509e662b08f249489ca48ebe8f174c9646514074064fd1985')
        tasks, review, inputs, old_audio = [read(p/'result.json')
                                           for p in (TASKS, REVIEW, INPUT, AUDIO)]
        if tasks['status'] != 'PASS_FIXED_DEVELOPMENT_TASKS':
            raise ValueError('complete frozen task corpus required')
        splits = {}
        for split in ('train', 'validation'):
            path = TASKS/(split+'.jsonl')
            pin(path, tasks['artifacts'][path.name])
            splits[split] = [json.loads(line) for line in path.read_text().splitlines()]
        events = frozen_events(splits)
        queries = query_inventory(splits)
        if (len(queries) != 24 or dict(Counter(x['split'] for x in events))
                != dict(train=8102, validation=3049)
                or {s: len(v) for s, v in splits.items()} != dict(train=23707, validation=8875)):
            raise ValueError('complete frozen task and query inventory required')
        reviews = {(r['parent'], r['event_row']+1): r for r in review['rows']}
        old_events = {(r['parent'], r['event_row']+1) for r in inputs['rows']}
        if args.phase == 'smoke':
            events = [e for e in events if (e['parent'], e['source_line']) in reviews]
            if len(events) != 21 or any(e['split'] != 'train' for e in events):
                raise ValueError('fixed 21 TRAIN smoke events required')
        for directory, data in ((REVIEW, review), (INPUT, inputs), (AUDIO, old_audio)):
            for name, digest in data['artifacts'].items():
                pin(directory/name, digest)
        for event in {e['parent']: e for e in events}.values():
            pin(MEDIA/event['media_file'], event['asset_sha256'])
        for name in subprocess.check_output(['git', 'ls-files', '-z', '--', '*.py'],
                                            cwd=REPO, text=True).split('\0'):
            if name:
                pin(REPO/name)
        audio_path = REPO/'configs/pretrained_reuse/native_audio_profile_v1.json'
        language_path = REPO/'configs/pretrained_reuse/native_language_cache_v1.json'
        pin(audio_path)
        pin(language_path)
        ac, lc = read(audio_path), read(language_path)
        speech_path, language_model = Path(ac['speech_model']), Path(lc['model_path'])
        for name in ('config.json', 'encoder.safetensors', 'preprocessor_config.json'):
            path = speech_path/name
            pin(path, ac['input_pins'][str(path)])
        pin(ac['event_weights'], ac['input_pins'][ac['event_weights']])
        for name, digest in lc['input_pins'].items():
            if Path(name).parent == language_model:
                pin(name, digest)
        source = Path(ac['event_source'])
        check_source(source, ac['event_revision'], 'models')
        for path in source.rglob('*.py'):
            pin(path)
        reference = receipt(args.reference, args.reference_sha256) if args.reference else None
        if reference and reference['phase'] != args.phase:
            raise ValueError('replay must use the same phase')
        smoke = None
        smoke_events = {}
        if args.phase == 'full':
            smoke = receipt(args.smoke, args.smoke_sha256)
            replay = receipt(args.smoke_replay, args.smoke_replay_sha256)
            if (smoke['phase'] != 'smoke' or replay['phase'] != 'smoke'
                    or replay['reference_sha256'] != args.smoke_sha256
                    or smoke['artifacts'] != replay['artifacts']):
                raise ValueError('exact complete smoke replay required before full cache')
            smoke_events = {e['event_id']: (p['parent'], e['bank_index'])
                            for p in smoke['parents'] for e in p['events']}
        write(args.output/'plan.json', dict(events=events, queries=queries))
        artifacts['plan.json'] = sha(args.output/'plan.json')
        result.update(expected_events=len(events), expected_queries=len(queries),
            expected_parents=len({e['parent'] for e in events}),
            expected_questions=sum(e['questions'] for e in events),
            expected_old_audio_matches=expected_reference_matches(events, old_events),
            speech_revision=ac['speech_revision'], event_revision=ac['event_revision'],
            language_revision=lc['model_revision'])
        import av
        import torch
        from safetensors.torch import load_file
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            WhisperConfig,
            WhisperFeatureExtractor,
        )
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        from scripts.capture_native_language import state_digest
        from tiny_omni_decision.native_language import candidate_vector

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('existing local CUDA environment required')
        torch.cuda.reset_peak_memory_stats()
        config = WhisperConfig.from_pretrained(speech_path, local_files_only=True)
        config._attn_implementation = 'eager'
        speech = WhisperEncoder(config)
        speech.load_state_dict(load_file(speech_path/'encoder.safetensors'), strict=True)
        speech.eval().requires_grad_(False).to('cuda')
        processor = WhisperFeatureExtractor.from_pretrained(speech_path, local_files_only=True)
        sys.path.insert(0, str(source))
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        with contextlib.redirect_stdout(io.StringIO()):
            event_model = get_model(width_mult=.4, pretrained_name=None, T_max=1.)
            mel = AugmentMelSTFT()
        event_model.load_state_dict(torch.load(ac['event_weights'], weights_only=True,
                                              map_location='cpu'), strict=True)
        event_model.eval().requires_grad_(False).to('cuda')
        mel.eval().requires_grad_(False).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(language_model, local_files_only=True)
        language, loading = AutoModelForSequenceClassification.from_pretrained(
            language_model, local_files_only=True, dtype=torch.float32,
            attn_implementation='eager', output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
            raise ValueError('language weights not loaded exactly')
        language.eval().requires_grad_(False).to('cuda')
        models = dict(speech=speech, event=event_model, mel=mel, language=language)
        result['parameters'] = {k: sum(p.numel() for p in v.parameters())
                                for k, v in models.items()}
        if result['parameters'] != dict(speech=8208384, event=1961239, mel=0, language=70830722):
            raise ValueError('fixed encoder inventory differs')

        def digest():
            if any(m.training for v in models.values() for m in v.modules()) or any(
                    p.requires_grad or p.dtype != torch.float32
                    for v in models.values() for p in v.parameters()):
                raise ValueError('all encoders must remain frozen eval FP32')
            return {k: state_digest(v) for k, v in models.items()}

        result.update(frozen_before=digest(), device=torch.cuda.get_device_name(),
            cuda=torch.version.cuda, python=platform.python_version(),
            ffmpeg_libraries=av.library_versions, versions={p: importlib.metadata.version(p)
                for p in ('torch', 'torchaudio', 'transformers', 'numpy', 'safetensors', 'av')})
        for prior in (reference, smoke):
            if prior and (prior['frozen_before'] != result['frozen_before']
                          or prior['versions'] != result['versions']):
                raise ValueError('reference encoder state/runtime differs')
        if any(result['frozen_before'][k] != old_audio['frozen_before'][k]
               for k in ('speech', 'event', 'mel')):
            raise ValueError('audio encoders differ from passing physical probe')
        query_tick = time.perf_counter()
        query_values = []
        for query in queries:
            guard()
            value = candidate_vector(language, tokenizer, query['question'], query['candidate'])
            query_values.append(value.detach().cpu().numpy())
        query_values = np.stack(query_values)
        result['query_forward_seconds'] = time.perf_counter()-query_tick
        if query_values.shape != (24, 384):
            raise ValueError('original paired candidate representation required')
        save('queries', query_values)
        for kind, positions in audio_positions().items():
            save(kind+'-positions', positions)
        if smoke:
            for name in ('queries.npy', 'speech-positions.npy', 'event_audio-positions.npy'):
                if artifacts[name] != smoke['artifacts'][name]:
                    raise ValueError('query/position representation differs from smoke')
        result['load_and_query_seconds'] = time.perf_counter()-tick
        write(args.output/'result.json', result)
        for parent, grouped in groupby(events, key=lambda e: e['parent']):
            group = list(grouped)
            path = MEDIA/group[0]['media_file']
            guard()
            parent_tick = time.perf_counter()
            chunks, first, previous_end = [], None, None
            with av.open(str(path)) as container:
                container.streams.audio[0].codec_context.thread_count = 1
                for frame in container.decode(audio=0):
                    stamp = float(frame.pts*frame.time_base)
                    if frame.sample_rate != 96000 or len(frame.layout.channels) != 1:
                        raise ValueError('native 96k mono required')
                    if previous_end is not None and abs(stamp-previous_end) > 1e-8:
                        raise ValueError('discontinuous audio PTS')
                    first = stamp if first is None else first
                    previous_end = stamp+frame.samples/frame.sample_rate
                    chunks.append(frame.to_ndarray().reshape(-1))
            pcm = np.concatenate(chunks)
            result['audio_decode_seconds'] += time.perf_counter()-parent_tick
            banks = dict(speech=np.full((len(group), 25, 384), np.nan, np.float32),
                         event_audio=np.full((len(group), 1, 384), np.nan, np.float32))
            records = []
            for index, event in enumerate(group):
                guard()
                crop, clock = crop_audio(pcm, 96000, first, event['event_seconds'])
                pcm16, pcm32 = resample_direct(crop, 16000), resample_direct(crop, 32000)
                key = (parent, event['source_line'])
                if key in reviews:
                    prefix = f"{reviews[key]['number']:02d}-{parent}"
                    np.testing.assert_array_equal(crop, np.load(REVIEW/(prefix+'-pcm96.npy')))
                    result['matched_review_events'] += 1
                if key in old_events:
                    for rate, values in ((16000, pcm16), (32000, pcm32)):
                        np.testing.assert_array_equal(values, np.load(
                            INPUT/(parent+f'-direct-{rate}.npy'), allow_pickle=False))
                forward_tick = time.perf_counter()
                speech_tokens, event_tokens = prepared_audio_tokens(
                    speech, processor, event_model, mel, pcm16, pcm32)
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter()-forward_tick
                result['audio_forward_seconds'] += forward_seconds
                values = dict(speech=speech_tokens.cpu().numpy(),
                              event_audio=event_tokens.cpu().numpy())
                store_audio_event(banks, index, values)
                if key in old_events:
                    for kind, value in values.items():
                        np.testing.assert_array_equal(value, np.load(
                            AUDIO/(parent+'-'+kind+'.npy'), allow_pickle=False))
                    result['matched_old_audio_events'] += 1
                if event['event_id'] in smoke_events:
                    ref_parent, ref_index = smoke_events[event['event_id']]
                    for kind, value in values.items():
                        np.testing.assert_array_equal(value, np.load(
                            args.smoke/(ref_parent+'-'+kind+'.npy'), allow_pickle=False)[ref_index])
                    result['matched_smoke_events'] += 1
                records.append(dict(**event, bank_index=index, timing=clock,
                    input_sha256={k: digest_array(v) for k, v in
                                  dict(pcm96=crop, pcm16=pcm16, pcm32=pcm32).items()},
                    feature_sha256={k: digest_array(v) for k, v in values.items()}))
                result['completed_events'] += 1
                result['completed_questions'] += event['questions']
            for kind, bank in banks.items():
                save(parent+'-'+kind, bank)
            if sha(path) != group[0]['asset_sha256']:
                raise ValueError('media changed during extraction')
            result['parents'].append(dict(parent=parent, split=group[0]['split'], events=records,
                                           seconds=time.perf_counter()-parent_tick))
            result['completed_parents'] += 1
            result['elapsed_seconds'] = time.perf_counter()-tick
            write(args.output/'result.json', result)
            print(result['completed_parents'], '/', result['expected_parents'],
                  result['completed_events'], '/', len(events), parent, flush=True)
        result['frozen_after'] = digest()
        if (result['frozen_after'] != result['frozen_before']
                or result['completed_events'] != len(events)
                or result['completed_questions'] != result['expected_questions']
                or result['completed_parents'] != result['expected_parents']
                or result['matched_review_events'] != 21
                or result['matched_old_audio_events'] != result['expected_old_audio_matches']
                or smoke and result['matched_smoke_events'] != 21):
            raise ValueError('incomplete coverage or encoder state changed')
        if reference:
            if artifacts != reference['artifacts']:
                raise ValueError('complete replay artifact set differs')
            for new, old in zip(result['parents'], reference['parents'], strict=True):
                if new['parent'] != old['parent'] or new['events'] != old['events']:
                    raise ValueError('replayed mapping, clocks or feature digests differ')
        check_source(source, ac['event_revision'], 'models')
        for path, expected in pins.items():
            if sha(Path(path)) != expected:
                raise ValueError('source/input changed while caching')
        if sha(args.output/'plan.json') != artifacts['plan.json']:
            raise ValueError('cache plan changed')
        guard()
        result['status'] = 'PASS_COMPLETE_AUDIO_QUERY_CACHE'
    except BaseException as exc:
        result.update(status='INCOMPLETE', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-tick, finished_utc=datetime.now(UTC).isoformat())
        write(args.output/'result.json', result)
        print(result['status'], flush=True)


if __name__ == '__main__':
    main()
