"""Eight fixed GPU readout updates per arm; mechanics/cost only, no quality selection."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import psutil

from scripts.greatest_hits_audio_admission import load_revalidated_audio
from scripts.greatest_hits_learning import backward_batch, smoke_plan
from scripts.greatest_hits_transport import bind_caches
from scripts.native_video_cache import sha
from scripts.probe_greatest_hits_transport import AUDIO, REPO, ROOT, TASKS, VIDEO, read, write

INITIAL = ROOT/'greatest-hits-transport-smoke-01'
ARMS = ('C0','E0G0','E0G1','E1G0','E1G1')
FULL_AUDIO = ROOT/'greatest-hits-audio-query-cache-01'
REVALIDATION = ROOT/'greatest-hits-audio-cache-revalidation-01/result.json'
REVALIDATION_SHA = '4b273e75298e640b79cbe5de0e234f1d1b6f54e45965af692cba86471b95f4cb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_sha256):
        raise ValueError('paired reference path/hash required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip():
        raise ValueError('clean committed source required')
    gpu = subprocess.check_output(['nvidia-smi',
        '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    used,utilization = [int(v.strip()) for v in gpu.split(',')]
    if used > 512 or utilization > 5:
        raise RuntimeError('GPU occupied; do not compete')
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    pins,artifacts = {},{}
    result = dict(status='RUNNING',command=[sys.executable,*sys.argv],pid=os.getpid(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        started_utc=datetime.now(UTC).isoformat(),gpu_preflight=gpu,
        encoder_updates=0,optimizer_updates=0,quality_evaluated=False,checkpoint_selection=False,
        validation_task_metadata_loaded=True,validation_feature_files_hashed=True,
        validation_examples_forwarded=0,sealed_audit_loaded=False,calibration_loaded=False,
        cloud_cost=0,pins=pins,artifacts=artifacts,arms={},reference_sha256=args.reference_sha256)
    torch = None

    def pin(path,expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('pin mismatch: '+str(path))
        pins[str(path)] = actual

    def guard():
        result['peak_working_set_bytes'] = psutil.Process().memory_info().peak_wset
        result['peak_vram_bytes'] = torch.cuda.max_memory_allocated() if torch else 0
        if (time.perf_counter()-started > 900 or result['peak_working_set_bytes'] > 6*2**30
                or result['peak_vram_bytes'] > 4*2**30
                or shutil.disk_usage(args.output).free < 16*2**30):
            raise RuntimeError('fixed learning smoke resource guard')

    try:
        guard()
        pin(INITIAL/'result.json',
            '7f6b60c0d4dfa40cb9a86096301f6e5757a0e57ab6b3975b19d74ae4db8167d3')
        pin(AUDIO/'result.json',
            '95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea')
        pin(VIDEO/'result.json',
            'd0ee01083f70a98be68b3a71c0a9abd68ac9b4b82759cc089e0232ff32bd38db')
        pin(TASKS/'result.json',
            '7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831')
        initial,audio,video,tasks = [read(p/'result.json') for p in (INITIAL,AUDIO,VIDEO,TASKS)]
        if initial['status'] != 'PASS_REAL_FEATURE_TRANSPORT':
            raise ValueError('passing real-feature transport required')
        for directory,receipt in ((INITIAL,initial),(AUDIO,audio),(VIDEO,video)):
            for name,digest in receipt['artifacts'].items():
                pin(directory/name,digest)
        pin(VIDEO/'plan.json',video['plan_sha256'])
        pin(REVALIDATION,REVALIDATION_SHA)
        admitted = load_revalidated_audio(FULL_AUDIO,REVALIDATION,REVALIDATION_SHA)
        result['audio_admission'] = admitted['audio_admission']
        pin(FULL_AUDIO/'result.json',admitted['audio_admission']['producer_result_sha256'])
        splits = {}
        for split in ('train','validation'):
            path = TASKS/(split+'.jsonl')
            pin(path,tasks['artifacts'][path.name])
            splits[split] = [json.loads(line) for line in path.read_text().splitlines()]
        binding = bind_caches(splits,audio,video,read(AUDIO/'plan.json'),
                              read(VIDEO/'plan.json'),phase='smoke')
        train = [r for r in splits['train'] if r['id'] in binding['questions']]
        if len(train) != 63 or len(binding['events']) != 21:
            raise ValueError('fixed21 TRAIN events and63 questions required')
        plan = smoke_plan(train)
        by_id = {r['id']:r for r in train}
        consumed = [by_id[x['id']] for b in plan for x in b]
        accounting = dict(consumed=32,unique_questions=len({r['id'] for r in consumed}),
            unique_events=len({r['event_id'] for r in consumed}),
            unique_assets=len({r['asset_sha256'] for r in consumed}),
            repeated_questions=32-len({r['id'] for r in consumed}),
            tasks=dict(Counter(r['task'] for r in consumed)),source='greatest-hits',split='train')
        config = dict(seed=17,updates=8,microbatches_per_update=4,optimizer='AdamW',
            lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=.01,clip_norm=1.,
            scheduler='constant smoke only',dtype='float32',arms=list(ARMS),accounting=accounting)
        for name in subprocess.check_output(['git','ls-files','-z','--','*.py'],cwd=REPO,
                                            text=True).split('\0'):
            if name:
                pin(REPO/name)
        reference = None
        if args.reference:
            pin(args.reference/'result.json',args.reference_sha256)
            reference = read(args.reference/'result.json')
            if (reference['status'] != 'PASS_AV_READOUT_LEARNING_SMOKE'
                    or reference['source_commit'] != result['source_commit']):
                raise ValueError('same-source successful learning smoke required')
            for name,digest in reference['artifacts'].items():
                pin(args.reference/name,digest)
        for name,value in (('plan.json',plan),('config.json',config)):
            write(args.output/name,value)
            artifacts[name] = sha(args.output/name)
        import torch

        from scripts.greatest_hits_transport_store import ExplicitEventStore
        from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
        from tiny_omni_decision.native_candidate import NativeCandidateReadout, candidate_json

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.cuda.is_available():
            raise RuntimeError('existing local CUDA required')
        torch.cuda.reset_peak_memory_stats()
        result.update(versions={p:importlib.metadata.version(p) for p in ('torch','numpy')},
            python=sys.version,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
            attention_backend_policy=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled(),
                cudnn=torch.backends.cuda.cudnn_sdp_enabled(),selection='PyTorch automatic'),
            tf32=False,deterministic=True,cpu_threads=4,config=config,
            load_seconds=time.perf_counter()-started)
        store = ExplicitEventStore(binding,dict(audio=AUDIO,video=VIDEO),
                                   dict(audio=audio['artifacts'],video=video['artifacts']),device='cuda')
        def create(name):
            return NativeCandidateReadout() if name == 'C0' else ConditionalAVReadout(
                exchange=name[1]=='1',gated=name[3]=='1')

        def inactive(name):
            base = '' if name == 'C0' else 'baseline.'
            names = [base+'projections.text.',base+'projections.image.']
            if name != 'C0':
                if name[1]=='0':
                    names += ['audio_from_video.','video_from_audio.','audio_norm.','video_norm.']
                if name[3]=='0':
                    names += ['gate.']
            return tuple(names)

        for arm in ARMS:
            guard()
            torch.manual_seed(17)
            model = create(arm)
            count = sum(p.numel() for p in model.parameters())
            if count != (889345 if arm == 'C0' else 1732420) or 195365529+count > 200000000:
                raise ValueError('measured encoder-inclusive parameter budget differs')
            state = torch.load(INITIAL/(arm+'.pt'),map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            before = {k:v.detach().clone() for k,v in model.named_parameters()}
            model.cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),
                                          eps=1e-8,weight_decay=.01)
            shutil.copyfile(INITIAL/(arm+'.pt'),args.output/(arm+'-initial.pt'))
            artifacts[arm+'-initial.pt'] = sha(args.output/(arm+'-initial.pt'))
            history,timing,max_groups = [],[],{}
            arm_started = time.perf_counter()
            for step,batch in enumerate(plan,1):
                guard()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                def forward(current,item):
                    blocks,q = store.get(item['id'],order=item['order'])
                    return current(blocks,q),item['target']
                losses = backward_batch(model,batch,forward)
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                groups = {}
                for name,p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise ValueError('nonfinite readout gradient')
                        key = name.split('.')[0]
                        groups.setdefault(key,[]).append(p.grad.detach().double().square().sum())
                norms = {k:float(torch.stack(v).sum().sqrt()) for k,v in groups.items()}
                for key,value in norms.items():
                    max_groups[key] = max(max_groups.get(key,0.),value)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                diagnostics_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError('nonfinite updated readout parameter')
                timing.append(dict(step=step,forward_backward_seconds=forward_seconds,
                    gradient_diagnostics_seconds=diagnostics_seconds,
                    optimizer_seconds=time.perf_counter()-tick))
                history.append(dict(step=step,losses=losses,mean_ce=sum(losses)/4,
                    gradient_norm=float(norm),group_gradient_norms=norms,lr=.001,
                    cumulative_examples=step*4))
                result['optimizer_updates'] += 1
            training_seconds = time.perf_counter()-arm_started
            changed = []
            for name,p in model.named_parameters():
                if not torch.equal(p.detach().cpu(),before[name]):
                    if name.startswith(inactive(arm)):
                        raise ValueError('unused or disabled parameter changed: '+name)
                    changed.append(name)
            required = ['projections','query','score'] if arm == 'C0' else [
                'baseline','audio_latents','video_latents','pool','candidate_attention','correction']
            if arm != 'C0' and arm[1]=='1':
                required += ['audio_from_video','video_from_audio','audio_norm','video_norm']
            if arm != 'C0' and arm[3]=='1':
                required += ['gate']
            if not changed or any(max_groups.get(k,0.)<=0 for k in required):
                raise ValueError('enabled readout path has no reached gradient')
            checkpoint = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                              step=8,arm=arm)
            final_path = args.output/(arm+'-final.pt')
            torch.save(checkpoint,final_path)
            artifacts[final_path.name] = sha(final_path)
            write(args.output/(arm+'-history.json'),history)
            artifacts[arm+'-history.json'] = sha(args.output/(arm+'-history.json'))
            model.eval()
            restored = create(arm).cuda().eval()
            restored.load_state_dict(torch.load(final_path,map_location='cpu',
                                                weights_only=True)['model'],strict=True)
            predictions = []
            inference_tick = time.perf_counter()
            with torch.inference_mode():
                for ident,row in binding['questions'].items():
                    for mode in ('joint','audio','video'):
                        blocks,q = store.get(ident,mode=mode)
                        scores = model(blocks,q)
                        torch.testing.assert_close(restored(blocks,q),scores,atol=0,rtol=0)
                        if arm != 'C0' and mode != 'joint':
                            torch.testing.assert_close(model.baseline(blocks,q),scores,atol=0,rtol=0)
                        predictions.append(dict(id=ident,mode=mode,logits=scores.tolist(),
                            decision=candidate_json(scores,row['options'])))
            torch.cuda.synchronize()
            write(args.output/(arm+'-predictions.json'),predictions)
            artifacts[arm+'-predictions.json'] = sha(args.output/(arm+'-predictions.json'))
            result['arms'][arm] = dict(updates=8,consumed=32,accounting=accounting,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                encoder_inclusive_parameters=195365529+sum(p.numel() for p in model.parameters()),
                changed_parameters=changed,max_group_gradient_norms=max_groups,
                training_seconds=training_seconds,timing=timing,
                prediction_seconds=time.perf_counter()-inference_tick,predictions=len(predictions))
            del model,restored,optimizer,checkpoint,state,before
            torch.cuda.empty_cache()
            write(args.output/'result.json',result)
            print(arm,'8 updates,189 reload-verified predictions',flush=True)
        store.verify_files()
        if result['optimizer_updates'] != 40 or set(result['arms']) != set(ARMS):
            raise ValueError('incomplete smoke updates')
        if reference and (reference['artifacts'] != artifacts or reference['versions']
                          != result['versions'] or reference['config'] != config):
            raise ValueError('fresh-process learning replay differs')
        for path,digest in pins.items():
            if sha(Path(path)) != digest:
                raise ValueError('input/source changed during learning smoke')
        guard()
        result['status'] = 'PASS_AV_READOUT_LEARNING_SMOKE'
    except BaseException as exc:
        result.update(status='INCOMPLETE',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter()-started,finished_utc=datetime.now(UTC).isoformat())
        write(args.output/'result.json',result)
        print(result['status'],flush=True)


if __name__ == '__main__':
    main()
