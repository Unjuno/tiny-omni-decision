"""Fixed16/32/64 SAM2 chunk matrix; numerical gates never relaxed."""

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from scripts.probe_dynamic_discovery import R1, WEIGHT_PIN, legacy_functions
from scripts.probe_timed_mask_tracking import require_arrays, verify_sources
from scripts.probe_track_appearance import ROOT, sha, write
from tiny_omni_decision.mask_trajectory_features import mask_query_points
from tiny_omni_decision.sam2_chunked_masks import (
    chunked_initial_mask_decode,
    extend_pinned_sources,
)

RAW_PREFIX_PIN = '73ebefe25631de792d0bc4f10e6492801b88383b12d49354c6bbbbbe045a901c'
SMOKE_PIN = 'bc05e706e702013b36a123166077e092442966f2e29ab01d785792bc55c83684'
FIXED_CHUNKS = (16, 32, 64)
FIXED_ORDER = (('native', 'chunk16', 'chunk32', 'chunk64'),
               ('chunk64', 'chunk32', 'chunk16', 'native'))
PRIOR_BATCH_PIN = '5122a01547fc8b1b2733b8582485ea2893b25b86511a75360cee0a461962856e'
PRIOR_BATCH_ARRAY_PIN = 'd8357e6ee0fd21a27e05daed05eeb64817c5eb1736a4de3dbb344aa58139d828'

RGB_PIN = 'b703c749ab94ba5b67e585f7ded3b7b68a9e785042f5e853fe135566c21fe554'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import av
    import torch
    import transformers
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    start = time.perf_counter()
    report = dict(status='PREFLIGHT', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), pid=os.getpid(),
        training=False, annotations_loaded=False, validation_opened=False,
        sealed_audit_opened=False, cloud_cost=0, windows=[])
    try:
        old = ROOT / 'dynamic-discovery-cache-smoke-01'
        if sha(old / 'result.json') != SMOKE_PIN:
            raise ValueError('pinned original raw cache smoke required')
        prior = json.loads((old / 'result.json').read_text())
        old_cfg = json.loads((old / 'config.json').read_text())
        if (prior['status'] != 'PASS_SAME_FRAME_CACHE_SMOKE'
                or sha(old / 'config.json') != prior['config_sha256']):
            raise ValueError('protected raw reference changed')
        plan = old_cfg['input_plan']
        if plan['video_id'] != 'video_10903' or plan['split'] != 'train':
            raise ValueError('fixed training prefix only')
        prefix_path = old / 'smoke-raw-prefix.npz'
        if sha(prefix_path) != RAW_PREFIX_PIN:
            raise ValueError('unchanged native first64 floating outputs required')
        with np.load(prefix_path, allow_pickle=False) as saved:
            archived = dict(raw=saved['0_raw_cached'].copy(), score=saved['0_score_cached'].copy())
        if archived['raw'].shape != (64, 256, 256) or archived['score'].shape != (64,):
            raise ValueError('complete original64 prefix schema required')
        prior_batch_path = ROOT / 'sam2-batched-decoder-prefix-01' / 'result.json'
        if sha(prior_batch_path) != PRIOR_BATCH_PIN:
            raise ValueError('preserved failed64 experiment required')
        prior_batch = json.loads(prior_batch_path.read_text())
        prior_batch_cfg = prior_batch_path.parent / 'effective-config.json'
        if (sha(prior_batch_cfg) != prior_batch['config_sha256']
                or prior_batch['status'] != 'PASS_MATCHED64_PROBE_NOT_DROPIN_EQUIVALENT'
                or sha(prior_batch['path']) != PRIOR_BATCH_ARRAY_PIN):
            raise ValueError('protected prior64 artifacts changed')
        hashes = dict(json.loads(prior_batch_cfg.read_text())['source_hashes'])
        verify_sources(hashes)
        hashes = extend_pinned_sources(hashes, {str(prior_batch_path): PRIOR_BATCH_PIN,
                       str(prior_batch_cfg): prior_batch['config_sha256'],
                       prior_batch['path']: PRIOR_BATCH_ARRAY_PIN,
                       str(Path('src/tiny_omni_decision/sam2_chunked_masks.py').resolve()):
                           sha(Path('src/tiny_omni_decision/sam2_chunked_masks.py'))})
        hashes = extend_pinned_sources(hashes, {
                       str(old / 'result.json'): SMOKE_PIN, str(prefix_path): RAW_PREFIX_PIN,
                       str(Path(__file__).resolve()): sha(Path(__file__)),
                       str(Path('src/tiny_omni_decision/sam2_batch_masks.py').resolve()):
                           sha(Path('src/tiny_omni_decision/sam2_batch_masks.py')),
                       str(Path('src/tiny_omni_decision/mask_trajectory_features.py').resolve()):
                           sha(Path('src/tiny_omni_decision/mask_trajectory_features.py'))})
        verify_sources(hashes)
        if (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3
                or shutil.disk_usage(args.output).free < 8 * 1024**3 + 256 * 1024**2):
            raise MemoryError('fixed12GiB GPU/8GiB-plus256MiB disk admission required')
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        model, info = Sam2VideoModel.from_pretrained(
            R1 / 'model', local_files_only=True, output_loading_info=True,
            dtype=torch.float32, attn_implementation='sdpa')
        load_keys = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
        if any(info.get(k) for k in load_keys):
            raise ValueError('strict pinned native load required')
        model.eval().requires_grad_(False).cuda()
        processor = Sam2VideoProcessor.from_pretrained(R1 / 'model', local_files_only=True)
        for cls in (type(model), type(processor), type(processor.video_processor),
                    type(model.prompt_encoder), type(model.mask_decoder)):
            path = Path(inspect.getfile(cls))
            hashes = extend_pinned_sources(hashes, {str(path): sha(path)})
        config = dict(source_hashes=hashes, weight_sha256=WEIGHT_PIN,
                      model_revision='de431c4043854a71d8101e17995dfe596bf101a5',
                      input_plan=plan, source_frame=0, prefix_start=0, objects=64,
                      dtype='float32', attention='sdpa', seed=17, tf32=False,
                      raw_abs_tolerance=1e-3, score_abs_tolerance=1e-5, relative_tolerance=0,
                      discrete_masks_boxes_seeds_nms_exact_required=True,
                      full1024_or_other_time_admission=False,
                      max_body_seconds=180, peak_torch_cap_bytes=8 * 1024**3,
                      limits_checked_after_calls_not_watchdogs=True,
                      chunk_sizes=list(FIXED_CHUNKS), measured_order=FIXED_ORDER,
                      scope='same first64 frame0; fixed16/32/64 chunks; no corpus replacement')
        write(args.output / 'effective-config.json', config)
        report['config_sha256'] = sha(args.output / 'effective-config.json')
        previous = None
        if args.reference:
            previous = json.loads((args.reference / 'result.json').read_text())
            old_config = args.reference / 'effective-config.json'
            if (previous['status'] != 'PASS_CHUNK_MATRIX_PREFIX_COMPLETE'
                    or sha(old_config) != previous['config_sha256']
                    or previous['config_sha256'] != report['config_sha256']
                    or sha(previous['path']) != previous['sha256']):
                raise ValueError('completed identical primary contract required before inference')
            report['reference_result_sha256'] = sha(args.reference / 'result.json')
        helpers = legacy_functions()
        report['state_before'] = helpers['state_sha'](model)
        report['parameters'] = sum(p.numel() for p in model.parameters())
        if report['parameters'] != 38962498:
            raise ValueError('frozen native model inventory changed')
        with av.open(plan['media_path']) as container:
            image = next(container.decode(video=0)).to_ndarray(format='rgb24')
        report['native_rgb_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
        if report['native_rgb_sha256'] != RGB_PIN:
            raise ValueError('fixed native first frame RGB changed')
        h, w = image.shape[:2]
        points = np.array([[(x + .5) * w / 32, (y + .5) * h / 32]
                           for y in range(32) for x in range(32)], np.float32)[:64]
        session = processor.init_video_session(video=image[None], inference_device='cuda',
                    processing_device='cpu', video_storage_device='cpu', dtype=torch.float32)
        pixels = session.processed_frames[0].clone()
        encoder_count = [0]

        def count_encoder(_module, _inputs):
            encoder_count[0] += 1

        hook = model.vision_encoder.register_forward_pre_hook(count_encoder)
        decoded_objects = []

        def count_decoder(_module, _inputs, kwargs):
            decoded_objects.append(int(kwargs['sparse_prompt_embeddings'].shape[1]))

        decoder_hook = model.mask_decoder.register_forward_pre_hook(count_decoder, with_kwargs=True)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]

        def add_points():
            session.reset_tracking_data()
            processor.add_inputs_to_inference_session(session, frame_idx=0,
                obj_ids=list(range(1, 65)), input_points=points[None, :, None].tolist(),
                input_labels=[[[1]] * 64])
            if session.get_obj_num() != 64:
                raise ValueError('every64 independent object must remain')

        def native():
            output = model(session, frame_idx=0, run_mem_encoder=False)
            return output.pred_masks[:, 0], output.object_score_logits.reshape(-1)

        def convert(pair):
            raw, score = [a.detach().cpu().numpy().copy() for a in pair]
            if (raw.shape != (64, 256, 256) or score.shape != (64,)
                    or raw.dtype != np.float32 or score.dtype != np.float32
                    or not np.isfinite(raw).all() or not np.isfinite(score).all()):
                raise ValueError('invalid native/candidate64 low-resolution outputs')
            return dict(raw=raw, score=score)

        def cap():
            if (time.perf_counter() - start > 180
                    or torch.cuda.max_memory_allocated() > 8 * 1024**3):
                raise RuntimeError('fixed resource admission exceeded; preserve incomplete attempt')

        arrays = {}
        with torch.inference_mode():
            add_points()
            first = convert(native())
            require_arrays(first, archived)
            if encoder_count[0] != 1:
                raise ValueError('first native call must encode exactly once')
            features, positions = model._prepare_vision_features(session, 0, 1)
            hi = [v.permute(1, 2, 0).view(1, v.size(2), *size)
                  for v, size in zip(features[:-1], model.backbone_feature_sizes[:-1], strict=True)]
            low = model._prepare_memory_conditioned_features(
                inference_session=session, frame_idx=0, obj_idx=0,
                is_initial_conditioning_frame=True, current_vision_features=features[-1],
                current_vision_positional_embeddings=positions[-1], num_total_frames=1)
            embeddings = hi + [low]
            q = torch.cat(
                [session.point_inputs_per_obj[i][0]['point_coords'] for i in range(64)], 1)
            labels = torch.cat(
                [session.point_inputs_per_obj[i][0]['point_labels'] for i in range(64)], 1)
            if q.shape != (1, 64, 1, 2) or labels.shape != (1, 64, 1):
                raise ValueError('unaltered native scaled prompt inventory required')
            embedding_clones = [a.clone() for a in embeddings]
            q_clone, label_clone = q.clone(), labels.clone()

            def chunked(size):
                return chunked_initial_mask_decode(model, embeddings, q, labels, size)

            # Native/head allocations warmed before alternating matched64 windows.
            add_points()
            warm_native = convert(native())
            warm_chunks = {f'chunk{size}': convert(chunked(size)) for size in FIXED_CHUNKS}
            require_arrays(warm_native, archived)
            cap()
            for rep, order in enumerate(config['measured_order']):
                for arm in order:
                    if arm == 'native':
                        add_points()  # setup excluded symmetrically from forward windows
                    count_before = encoder_count[0]
                    decode_before = len(decoded_objects)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    output = (native() if arm == 'native'
                              else chunked(int(arm.removeprefix('chunk'))))
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - tick
                    value = convert(output)
                    del output
                    cap()
                    if encoder_count[0] != count_before:
                        raise ValueError('measured windows must reuse the same frame embeddings')
                    sizes = decoded_objects[decode_before:]
                    expected_sizes = ([1] * 64 if arm == 'native'
                                      else [int(arm.removeprefix('chunk'))]
                                      * (64 // int(arm.removeprefix('chunk'))))
                    if sizes != expected_sizes or sum(sizes) != 64:
                        raise ValueError('every window must actually decode every ordered64 object')
                    for key, a in value.items():
                        arrays[f'{arm}_{rep}_{key}'] = a
                    report['windows'].append(dict(arm=arm, rep=rep, objects=64, seconds=seconds,
                                                   decoder_object_counts=sizes,
                                                   extra_encoder_calls=0))
            if not all(torch.equal(a, b)
                       for a, b in zip(embeddings, embedding_clones, strict=True)):
                raise ValueError('batched decoding mutated source embeddings')
            if not torch.equal(q, q_clone) or not torch.equal(labels, label_clone):
                raise ValueError('batched decoding mutated native prompts')
        hook.remove()
        decoder_hook.remove()
        require_arrays(dict(raw=arrays['native_0_raw'], score=arrays['native_0_score']), archived)
        require_arrays(dict(raw=arrays['native_1_raw'], score=arrays['native_1_score']), archived)
        report['parity'] = {}
        native_masks = archived['raw'] > 0
        native_boxes = helpers['boxes_from_masks'](native_masks)
        native_nms = np.asarray(helpers['nms'](native_boxes, archived['score']), np.int64)
        native_seed, native_valid = mask_query_points(native_masks)
        for size in FIXED_CHUNKS:
            arm = f'chunk{size}'
            candidate = dict(raw=arrays[f'{arm}_0_raw'], score=arrays[f'{arm}_0_score'])
            require_arrays(candidate,
                           dict(raw=arrays[f'{arm}_1_raw'], score=arrays[f'{arm}_1_score']))
            require_arrays(warm_chunks[arm], candidate)
            masks = candidate['raw'] > 0
            boxes = helpers['boxes_from_masks'](masks)
            nms = np.asarray(helpers['nms'](boxes, candidate['score']), np.int64)
            seed, valid = mask_query_points(masks)
            parity = dict(
                raw_byte_exact=archived['raw'].tobytes() == candidate['raw'].tobytes(),
                score_byte_exact=archived['score'].tobytes() == candidate['score'].tobytes(),
                raw_max_abs=float(np.abs(archived['raw'].astype(np.float64)
                                         - candidate['raw'].astype(np.float64)).max()),
                score_max_abs=float(np.abs(archived['score'].astype(np.float64)
                                           - candidate['score'].astype(np.float64)).max()),
                changed_mask_pixels=int((native_masks != masks).sum()),
                changed_proposal_masks=int((native_masks != masks).any(axis=(1, 2)).sum()),
                boxes_byte_exact=native_boxes.tobytes() == boxes.tobytes(),
                prefix_nms_exact=native_nms.tobytes() == nms.tobytes(),
                foreground_seeds_exact=native_seed.tobytes() == seed.tobytes(),
                nonempty_flags_exact=native_valid.tobytes() == valid.tobytes())
            parity['prefix_only_admission'] = bool(
                parity['raw_max_abs'] <= config['raw_abs_tolerance']
                and parity['score_max_abs'] <= config['score_abs_tolerance']
                and parity['changed_mask_pixels'] == 0
                and all(parity[k] for k in ('boxes_byte_exact', 'prefix_nms_exact',
                                           'foreground_seeds_exact', 'nonempty_flags_exact')))
            report['parity'][arm] = parity
        with np.load(prior_batch['path'], allow_pickle=False) as saved:
            require_arrays(dict(raw=arrays['chunk64_0_raw'], score=arrays['chunk64_0_score']),
                           dict(raw=saved['batch_0_raw'], score=saved['batch_0_score']))
        report['prior_failed64_reproduced_bitwise'] = True
        dest = args.output / 'matched64-outputs.npz'
        np.savez_compressed(dest, **arrays)
        with np.load(dest, allow_pickle=False) as saved:
            require_arrays(saved, arrays)
        if args.reference:
            if (sha(args.reference / 'result.json') != report['reference_result_sha256']
                    or sha(previous['path']) != previous['sha256']):
                raise ValueError('protected primary changed during fresh replay')
            with np.load(previous['path'], allow_pickle=False) as saved:
                require_arrays(saved, arrays)
            report['fresh_replay_bitwise'] = True
        report['state_after'] = helpers['state_sha'](model)
        if report['state_before'] != report['state_after']:
            raise ValueError('frozen model state changed')
        if not all(not p.requires_grad and p.grad is None for p in model.parameters()):
            raise ValueError('frozen parameter flags or gradients changed')
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(
                torch.equal(a, b)
                for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)):
            raise ValueError('native/batched inference unexpectedly consumed RNG')
        if not torch.equal(session.processed_frames[0], pixels):
            raise ValueError('protected native RGB preprocessing changed')
        verify_sources(hashes)
        cap()
        admitted_chunks = [size for size in FIXED_CHUNKS
                           if report['parity'][f'chunk{size}']['prefix_only_admission']]
        report.update(status='PASS_CHUNK_MATRIX_PREFIX_COMPLETE',
                      prefix_admitted_chunks=admitted_chunks,
                      full_corpus_or_quality_admission=False,
                      path=str(dest), sha256=sha(dest), seconds=time.perf_counter() - start,
                      encoder_calls_total=encoder_count[0], rng_unchanged=True,
                      peak_torch_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      transformers=transformers.__version__)
        write(args.output / 'result.json', report)
        print(report['status'], report['seconds'], report['peak_torch_bytes'],
              json.dumps(report['parity']), flush=True)
    except Exception:
        report.update(status='FAIL_INCOMPLETE', seconds=time.perf_counter() - start,
                      error=traceback.format_exc())
        write(args.output / 'result.json', report)
        raise


if __name__ == '__main__':
    main()
