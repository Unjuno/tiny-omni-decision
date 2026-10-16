"""Independently check all saved audio inputs/banks after the known reference-count error.

Preserves the failed producer receipt. No encoder replay, training or status rewrite.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from scripts.greatest_hits_audio_query import expected_reference_matches, query_inventory
from scripts.greatest_hits_cache import frozen_events
from scripts.greatest_hits_inputs import crop_audio, resample_direct
from scripts.native_video_cache import sha

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')
CACHE = ROOT/'greatest-hits-audio-query-cache-01'
CACHE_SHA = '74a12e2d2779751eed6867b3fd0186e6ad14d9ea44a527a2d9b0ceee41a1c10d'
SMOKE = ROOT/'greatest-hits-audio-query-smoke-01'
PROBE = ROOT/'greatest-hits-input-physical-01'
TASKS = ROOT/'greatest-hits-development-tasks-01'
MEDIA = ROOT/'greatest-hits-development-media-01/members'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=REPO).strip()):
        raise ValueError('committed clean source and CPU-only verification required')
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    pins, artifacts = {}, {}
    result = dict(status='RUNNING', command=[sys.executable,*sys.argv], pid=os.getpid(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,
                                             text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(), source_cache=str(CACHE),
        producer_result_sha256=CACHE_SHA, producer_status='INCOMPLETE',
        encoder_replayed=False, optimizer_updates=0, quality_evaluated=False,
        validation_media_loaded=True, sealed_audit_loaded=False, calibration_loaded=False,
        cloud_cost=0, verified_events=0, verified_parents=0, verified_questions=0,
        input_replay_matches=0, feature_digest_matches=0, smoke_feature_matches=0,
        pins=pins, artifacts=artifacts)

    def pin(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('hash mismatch: '+str(path))
        pins[str(path)] = actual

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        if (time.perf_counter()-started > 900 or result['peak_working_set_bytes'] > 4*2**30
                or shutil.disk_usage(args.output).free < 16*2**30):
            raise RuntimeError('fixed CPU verification resource guard')

    try:
        guard()
        pin(CACHE/'result.json', CACHE_SHA)
        pin(SMOKE/'result.json',
            '95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea')
        pin(PROBE/'result.json',
            '97c91689effda1e7a080b81ddd4be7738718168493137a565f821371d493c0f6')
        pin(TASKS/'result.json',
            '7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831')
        original, smoke, probe, tasks = [read(p/'result.json')
                                       for p in (CACHE,SMOKE,PROBE,TASKS)]
        if (original['status'] != 'INCOMPLETE' or original['phase'] != 'full'
                or original['source_commit'] != '61530adc76935e4b5dbfbabf582ef1ec03cd1405'
                or original['error'] != "ValueError('incomplete coverage or encoder state changed')"
                or original['frozen_before'] != original['frozen_after']
                or original['frozen_before'] != smoke['frozen_before']
                or original['versions'] != smoke['versions'] or original['optimizer_updates'] != 0):
            raise ValueError('only the exactly pinned terminal counter failure is admissible')
        for name in subprocess.check_output(['git','ls-files','-z','--','*.py'], cwd=REPO,
                                            text=True).split('\0'):
            if name:
                pin(REPO/name)
        for path, digest in original['pins'].items():
            pin(path,digest)
        for name,digest in original['artifacts'].items():
            pin(CACHE/name,digest)
            artifacts[str(CACHE/name)] = digest
        artifacts[str(CACHE/'result.json')] = CACHE_SHA
        if len(artifacts) != 823:
            raise ValueError('complete823-file produced inventory required')
        splits = {}
        for split in ('train','validation'):
            path = TASKS/(split+'.jsonl')
            pin(path,tasks['artifacts'][path.name])
            splits[split] = [json.loads(line) for line in path.read_text().splitlines()]
        events = frozen_events(splits)
        plan = read(CACHE/'plan.json')
        if plan != dict(events=events, queries=query_inventory(splits)):
            raise ValueError('producer plan is not complete frozen corpus')
        for key,n in dict(events=11151,parents=409,questions=32582).items():
            for prefix in ('expected','completed'):
                if original[prefix+'_'+key] != n:
                    raise ValueError('producer coverage mismatch')
        references = {(r['parent'],r['event_row']+1) for r in probe['rows']}
        included = sorted(references.intersection((e['parent'],e['source_line']) for e in events))
        excluded = sorted(references.difference(included))
        if (expected_reference_matches(events,references) != 2
                or original['matched_old_audio_events'] != len(included)
                or excluded != [('2015-03-30-00-26-27',1)]
                or original['matched_review_events'] != 21
                or original['matched_smoke_events'] != 21):
            raise ValueError('known reference intersection differs')
        annotation = ROOT/'greatest-hits-train-inventory-01/members/2015-03-30-00-26-27_times.txt'
        pin(annotation,'f591880d4f61c2f4764c8bc0330cfb3247878f466f2cef43c19d7ffa86872750')
        omitted_line = annotation.read_text().splitlines()[0].split()
        if omitted_line != ['1.778125','None','None','None']:
            raise ValueError('excluded reference is not the documented all-missing-label event')
        result.update(reference_included=included, reference_excluded=excluded,
            original_expected_reference_matches=3, corrected_expected_reference_matches=2,
            omitted_reference_source_line=omitted_line,
            versions={p: importlib.metadata.version(p) for p in ('numpy','av')})
        if result['versions'] != {p: original['versions'][p] for p in ('numpy','av')}:
            raise ValueError('native input replay runtime differs from producer')
        for name in ('queries.npy','speech-positions.npy','event_audio-positions.npy'):
            pin(SMOKE/name,smoke['artifacts'][name])
            if sha(CACHE/name) != sha(SMOKE/name):
                raise ValueError('query or shared position cache differs')
        smoke_map = {e['event_id']:(p['parent'],e['bank_index'])
                     for p in smoke['parents'] for e in p['events']}
        offset = 0
        import av
        for parent in original['parents']:
            guard()
            group = parent['events']
            expected = events[offset:offset+len(group)]
            if (not group or parent['parent'] != expected[0]['parent']
                    or parent['split'] != expected[0]['split']):
                raise ValueError('parent order/binding differs')
            path = MEDIA/expected[0]['media_file']
            pin(path,expected[0]['asset_sha256'])
            chunks, first, previous_end = [],None,None
            with av.open(str(path)) as container:
                container.streams.audio[0].codec_context.thread_count = 1
                for frame in container.decode(audio=0):
                    stamp = float(frame.pts*frame.time_base)
                    if (frame.sample_rate != 96000 or len(frame.layout.channels) != 1
                            or previous_end is not None and abs(stamp-previous_end)>1e-8):
                        raise ValueError('native audio clock changed')
                    first = stamp if first is None else first
                    previous_end = stamp+frame.samples/96000
                    chunks.append(frame.to_ndarray().reshape(-1))
            pcm = np.concatenate(chunks)
            banks = {kind: np.load(CACHE/(parent['parent']+'-'+kind+'.npy'),
                                   mmap_mode='r',allow_pickle=False)
                     for kind in ('speech','event_audio')}
            for kind,count in (('speech',25),('event_audio',1)):
                if (banks[kind].shape != (len(group),count,384)
                        or banks[kind].dtype != np.float32 or not np.isfinite(banks[kind]).all()):
                    raise ValueError('native feature bank incomplete/nonfinite')
            for index,(actual,event) in enumerate(zip(group,expected,strict=True)):
                if (any(actual.get(k)!=v for k,v in event.items()) or actual['bank_index']!=index):
                    raise ValueError('saved event metadata differs from frozen task identity')
                crop,clock = crop_audio(pcm,96000,first,event['event_seconds'])
                if clock != actual['timing']:
                    raise ValueError('redecoded physical clock differs')
                values = dict(pcm96=crop, pcm16=resample_direct(crop,16000),
                              pcm32=resample_direct(crop,32000))
                if {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in values.items()
                        } != actual['input_sha256']:
                    raise ValueError('fresh native decode/resample input replay differs')
                result['input_replay_matches'] += 1
                for kind,bank in banks.items():
                    if hashlib.sha256(bank[index].tobytes()).hexdigest() != (
                            actual['feature_sha256'][kind]):
                        raise ValueError('saved features differ from produced tensors')
                    result['feature_digest_matches'] += 1
                    if event['event_id'] in smoke_map:
                        p,i = smoke_map[event['event_id']]
                        smoke_path = SMOKE/(p+'-'+kind+'.npy')
                        pin(smoke_path,smoke['artifacts'][smoke_path.name])
                        np.testing.assert_array_equal(bank[index],np.load(smoke_path)[i])
                        result['smoke_feature_matches'] += 1
                result['verified_events'] += 1
                result['verified_questions'] += event['questions']
            offset += len(group)
            result['verified_parents'] += 1
            result['elapsed_seconds'] = time.perf_counter()-started
            write(args.output/'result.json',result)
            print(result['verified_parents'],'/409',result['verified_events'],'/11151',flush=True)
        if (result['verified_events'] != 11151 or result['input_replay_matches'] != 11151
                or result['feature_digest_matches'] != 22302
                or result['smoke_feature_matches'] != 42
                or result['verified_parents'] != 409 or result['verified_questions'] != 32582):
            raise ValueError('revalidation coverage incomplete')
        for path,digest in pins.items():
            if sha(Path(path)) != digest:
                raise ValueError('input/source changed during revalidation')
        write(args.output/'artifact-hashes.json',artifacts)
        result['artifact_ledger_sha256'] = sha(args.output/'artifact-hashes.json')
        guard()
        result['status'] = 'PASS_FULL_AUDIO_CACHE_REVALIDATION'
    except BaseException as exc:
        result.update(status='INCOMPLETE',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-started,finished_utc=datetime.now(UTC).isoformat())
        write(args.output/'result.json',result)
        print(result['status'],flush=True)


if __name__ == '__main__':
    main()
