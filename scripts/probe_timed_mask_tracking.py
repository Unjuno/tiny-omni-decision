"""Fixed train-only timed model-mask queries; no source annotations or fitting."""

import argparse
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

import numpy as np

from scripts.probe_track_appearance import ROOT, sha, write
from scripts.score_dynamic_discovery import validate_proposal_bank
from tiny_omni_decision.mask_trajectory_features import (
    query_observation_support,
    timed_mask_queries,
)
from tiny_omni_decision.trajectory_features import trajectory_features

DISCOVERY_PIN = '191796c3e85ffea628c55d45f15cd1a09ecdba546551476e57dd55ce09a7ae0e'
TAPIR_LEDGER_PIN = '30910583657724619e82334fd7effca128b9e5415834422c8f4804c75ffe3d68'
R1 = Path('C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01')


def verify_sources(hashes):
    for path, pin in hashes.items():
        if sha(path) != pin:
            raise ValueError('protected input changed: ' + path)


def reconstruct_queries():
    """Rebuild every seed from immutable original discovery banks, no saved-plan trust."""
    run = ROOT / 'dynamic-discovery-fixed32-01'
    result_path = run / 'result.json'
    if sha(result_path) != DISCOVERY_PIN:
        raise ValueError('fixed complete discovery result required')
    result = json.loads(result_path.read_text())
    config_path = run / 'config.json'
    if sha(config_path) != result['config_sha256']:
        raise ValueError('fixed discovery config required')
    config = json.loads(config_path.read_text())
    plan = config['input_plan']
    if (
        result['status'] != 'PASS_FIXED32_DYNAMIC_PROPOSALS_NOT_QA'
        or plan['split'] != 'train' or plan['video_id'] != 'video_10903'
        or plan['frames'] != plan['cut_frame'] or plan['frames'] != 558
        or [row['frame_index'] for row in result['rows']] != plan['frame_indices']
        or plan['frame_indices'] != np.rint(np.linspace(0, 557, 32)).astype(int).tolist()
    ):
        raise ValueError('fixed identity/cut/all32 times required')
    hashes = dict(config['source_hashes'])
    hashes[str(result_path)] = DISCOVERY_PIN
    hashes[str(config_path)] = sha(config_path)
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    helper = Path('src/tiny_omni_decision/mask_trajectory_features.py').resolve()
    hashes[str(helper)] = sha(helper)
    hashes[plan['media_path']] = plan['media_sha256']
    for row in result['rows']:
        hashes[row['path']] = row['sha256']
    verify_sources(hashes)
    blocks, ids, flags, rows = [], [], [], []
    for row in result['rows']:
        with np.load(row['path'], allow_pickle=False) as saved:
            validate_proposal_bank(saved)
            nms = saved['nms_indices']
            masks = np.unpackbits(saved['masks_packed'][nms], axis=1, bitorder='big')
            masks = masks.reshape(len(nms), 256, 256).astype(bool)
            q, identity, valid = timed_mask_queries(
                masks, nms, frame_index=row['frame_index'], frames=plan['frames'])
        blocks.append(q)
        ids.append(identity)
        flags.append(valid)
        rows.append(dict(frame_index=row['frame_index'], offset=sum(len(b) for b in blocks[:-1]),
                         count=len(q), nonempty=int(valid.sum()), bank_sha256=row['sha256']))
    query = np.concatenate(blocks)
    indices = np.concatenate(ids)
    valid = np.concatenate(flags)
    if len(query) != 1070 or not valid.all():
        raise ValueError('all1070 nonempty NMS proposals required; no silent filtering')
    times = np.asarray(plan['frame_indices'], np.int64)
    support = query_observation_support(query, times)
    arrays = dict(query_tyx=query, proposal_indices=indices, nonempty=valid,
                  frame_indices=times, on_or_after_observation=support)
    verify_sources(hashes)
    return dict(input_plan=plan, source_hashes=hashes, rows=rows), arrays


def plan_queries(output):
    start = time.perf_counter()
    contract, arrays = reconstruct_queries()
    query = arrays['query_tyx']
    dest = output / 'queries.npz'
    np.savez_compressed(dest, **arrays)
    with np.load(dest, allow_pickle=False) as saved:
        require_arrays(saved, arrays)
    verify_sources(contract['source_hashes'])
    report = dict(status='PASS_ALL_TIMED_MODEL_MASK_QUERIES_NOT_TRACKING',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                        text=True).strip(),
                  **contract, queries=len(query),
                  unique_time_xy=len(np.unique(query, axis=0)),
                  observations_not_object_birth=True, no_candidate_dropped=True,
                  queries_path=str(dest), queries_sha256=sha(dest),
                  annotations_loaded=False, training=False, sealed_audit_opened=False,
                  seconds=time.perf_counter() - start)
    write(output / 'result.json', report)
    print(report['status'], report['queries'], flush=True)


def require_arrays(saved, arrays):
    if set(saved) != set(arrays):
        raise ValueError('array inventory changed')
    for key, value in arrays.items():
        actual = saved[key]
        if (actual.shape != value.shape or actual.dtype != value.dtype
                or actual.tobytes() != value.tobytes()):
            raise ValueError('array shape/dtype/bytes mismatch: ' + key)


def require_bundle_binding(plan, saved, contract, arrays):
    for key in ('input_plan', 'source_hashes', 'rows'):
        if plan.get(key) != contract[key]:
            raise ValueError('saved query plan differs from pinned discovery: ' + key)
    require_arrays(saved, arrays)


def track(output, query_run, reference):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import av
    import torch

    sys.path[:0] = [str(R1 / 'deps'), str(R1 / 'source')]
    from tapnet.torch.tapir_model import TAPIR

    start = time.perf_counter()
    report = dict(status='PREFLIGHT', pid=os.getpid(),
                  started_utc=datetime.now(UTC).isoformat(),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                        text=True).strip(),
                  annotations_loaded=False, training=False, cloud_cost=0)
    try:
        query_result = query_run / 'result.json'
        plan = json.loads(query_result.read_text())
        if plan['status'] != 'PASS_ALL_TIMED_MODEL_MASK_QUERIES_NOT_TRACKING':
            raise ValueError('complete fixed query plan required')
        contract, expected_queries = reconstruct_queries()
        hashes = dict(plan['source_hashes'])
        hashes[str(query_result)] = sha(query_result)
        hashes[plan['queries_path']] = plan['queries_sha256']
        ledger_path = ROOT / 'perception-occlusion-trajectory-features-01/result.json'
        if sha(ledger_path) != TAPIR_LEDGER_PIN:
            raise ValueError('pinned frozen tracker ledger required')
        ledger = json.loads(ledger_path.read_text())
        hashes.update(ledger['source_hashes'])
        hashes[str(ledger_path)] = TAPIR_LEDGER_PIN
        weight = R1 / 'tapir_checkpoint_panning.pt'
        hashes[str(weight)] = ledger['weight_sha256']
        converter = Path('src/tiny_omni_decision/trajectory_features.py').resolve()
        hashes[str(converter)] = sha(converter)
        verify_sources(hashes)
        free, _ = torch.cuda.mem_get_info()
        if free < 14 * 1024**3 or shutil.disk_usage(output).free < 8 * 1024**3:
            raise MemoryError('fixed GPU/disk admission reserve unavailable')
        with np.load(plan['queries_path'], allow_pickle=False) as saved:
            require_bundle_binding(plan, saved, contract, expected_queries)
            queries_np = saved['query_tyx'].copy()
            ids = saved['proposal_indices'].copy()
            times = saved['frame_indices'].copy()
            support = saved['on_or_after_observation'].copy()
            if not saved['nonempty'].all() or len(queries_np) != 1070:
                raise ValueError('all1070 real mask seeds required')
        config = dict(source_hashes=hashes, query_plan_sha256=sha(query_result),
                      seed=17, query_chunk_size=16, dtype='float32', tf32=False,
                      pyramid_level=0, extra_convs=False, input_plan=plan['input_plan'],
                      peak_torch_cap_bytes=12 * 1024**3, max_body_seconds=600,
                      limits_checked_after_forward_not_watchdogs=True,
                      offline_complete_video=True, training=False)
        write(output / 'effective-config.json', config)
        report['config_sha256'] = sha(output / 'effective-config.json')
        previous = None
        if reference:
            previous = json.loads((reference / 'result.json').read_text())
            if previous['status'] != 'PASS_ALL_TIMED_TRACKS_NOT_QA':
                raise ValueError('completed primary required for fresh replay')
            old_config_path = reference / 'effective-config.json'
            if sha(old_config_path) != previous['config_sha256']:
                raise ValueError('protected primary config changed')
            if json.loads(old_config_path.read_text()) != config:
                raise ValueError('fresh replay effective config differs from primary')
            if sha(previous['path']) != previous['sha256']:
                raise ValueError('protected primary bank changed')
            report['reference_result_sha256'] = sha(reference / 'result.json')
            report['reference_config_sha256'] = sha(old_config_path)
            report['reference_path'] = str(reference)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        model = TAPIR(pyramid_level=0, extra_convs=False).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(weight, map_location='cpu', weights_only=True), strict=True)
        model.cuda()
        from scripts.probe_dynamic_discovery import legacy_functions

        state_sha = legacy_functions()['state_sha']
        report['state_before'] = state_sha(model)
        report['parameters'] = sum(p.numel() for p in model.parameters())
        if report['parameters'] != 31072327:
            raise ValueError('frozen tracker parameter inventory changed')
        frames = []
        with av.open(plan['input_plan']['media_path']) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i >= plan['input_plan']['cut_frame']:
                    break
                frames.append(frame.reformat(width=256, height=256, format='rgb24').to_ndarray())
        if len(frames) != 558:
            raise ValueError('original full cut required')
        video = torch.from_numpy(np.stack(frames)).float()[None].cuda() / 255 * 2 - 1
        queries = torch.from_numpy(queries_np.copy())[None].cuda()
        report.update(status='RUNNING', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cuda=torch.version.cuda, python=platform.python_version(),
                      platform=platform.platform(), numpy=np.__version__, av=av.__version__,
                      loader_backend='native PyTorch CUDA on Windows',
                      preflight_free_bytes=free)
        write(output / 'progress.json', report)
        print('RUNNING', len(queries_np), len(frames), flush=True)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        with torch.inference_mode():
            predicted = model(video, queries, query_chunk_size=16)
            torch.cuda.synchronize()
            report['forward_seconds'] = time.perf_counter() - forward_start
            cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            # Full-video query sampling uses each original time, never video[:,:1].
            appearance = model.get_query_features(video, False, queries)
            app = torch.cat([appearance.lowres[0], appearance.hires[0]], -1)[0].cpu().numpy()
            arrays = {k: predicted[k][0].cpu().numpy()
                      for k in ('tracks', 'occlusion', 'expected_dist')}
            cpu_final, cuda_final = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        if not torch.equal(cpu_after, cpu_final) or not all(
                torch.equal(a, b) for a, b in zip(cuda_after, cuda_final, strict=True)):
            raise ValueError('query appearance extraction unexpectedly consumed RNG')
        if not torch.equal(queries[0].cpu(), torch.from_numpy(queries_np)):
            raise ValueError('model mutated time-specific query coordinates')
        if arrays['tracks'].shape != (1070, 558, 2) or app.shape != (1070, 384):
            raise ValueError('full query/time/appearance inventory changed')
        if not all(np.isfinite(a).all() for a in [app, *arrays.values()]):
            raise ValueError('nonfinite tracker result')
        geometry, retained = trajectory_features(**arrays)
        if not np.array_equal(retained, times):
            raise ValueError('original32 retained times changed')
        arrays.update(query_tyx=queries_np, proposal_indices=ids, frame_indices=times,
                      on_or_after_observation=support, query_appearance=app,
                      tokens=np.concatenate([geometry, app], -1),
                      cpu_rng_before=cpu_before.numpy(), cpu_rng_after=cpu_after.numpy(),
                      cpu_rng_final=cpu_final.numpy(),
                      cuda_rng_before=np.stack([r.cpu().numpy() for r in cuda_before]),
                      cuda_rng_after=np.stack([r.cpu().numpy() for r in cuda_after]),
                      cuda_rng_final=np.stack([r.cpu().numpy() for r in cuda_final]))
        dest = output / 'video_10903.npz'
        np.savez_compressed(dest, **arrays)
        with np.load(dest, allow_pickle=False) as saved:
            require_arrays(saved, arrays)
        if reference:
            if sha(previous['path']) != previous['sha256']:
                raise ValueError('protected primary bank changed')
            with np.load(previous['path'], allow_pickle=False) as saved:
                require_arrays(saved, arrays)
            report['fresh_replay_all_arrays_bitwise'] = True
            if sha(reference / 'result.json') != report['reference_result_sha256']:
                raise ValueError('protected primary result changed during replay')
        report['state_after'] = state_sha(model)
        if report['state_before'] != report['state_after']:
            raise ValueError('frozen tracker state changed')
        if not all(not p.requires_grad and p.grad is None for p in model.parameters()):
            raise ValueError('frozen tracker flags/gradients changed')
        verify_sources(hashes)
        peak = torch.cuda.max_memory_allocated()
        seconds = time.perf_counter() - start
        if peak > config['peak_torch_cap_bytes'] or seconds > config['max_body_seconds']:
            raise RuntimeError('resource admission exceeded; preserve incomplete result')
        report.update(status='PASS_ALL_TIMED_TRACKS_NOT_QA', path=str(dest), sha256=sha(dest),
                      query_count=1070, original_frames=558, retained_times=32,
                      file_bytes=dest.stat().st_size, arrays_readback_bitwise=True,
                      peak_torch_bytes=peak, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      tracker_cpu_rng_consumed=not torch.equal(cpu_before, cpu_after),
                      appearance_rng_unchanged=True,
                      seconds=seconds, ended_utc=datetime.now(UTC).isoformat())
        write(output / 'result.json', report)
        print(report['status'], seconds, peak, flush=True)
    except Exception:
        report.update(status='FAIL_INCOMPLETE', error=traceback.format_exc(),
                      seconds=time.perf_counter() - start)
        write(output / 'result.json', report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-run', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output directory required')
    args.output.mkdir(parents=True)
    if args.query_run:
        track(args.output, args.query_run, args.reference)
    else:
        if args.reference:
            raise ValueError('fresh replay requires fixed query run')
        plan_queries(args.output)


if __name__ == '__main__':
    main()
