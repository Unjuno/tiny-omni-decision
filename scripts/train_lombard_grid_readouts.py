"""One fixed three-arm local comparison; --smoke evaluates training sources only."""

import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.clap_progress_io import save_progress
from scripts.lombard_readout_core import (
    OPTIONS,
    SIZES,
    SlotHead,
    fit_normalizer,
    json_decision,
    make_inputs,
    select_temperatures,
    selected_epoch,
    summarize,
)
from scripts.lombard_readout_data import load_data, sha

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')


def sample_orders(count, epochs):
    if count < 1 or epochs < 1:
        raise ValueError('positive sampler budget required')
    generator = torch.Generator().manual_seed(17)
    return torch.stack([torch.randperm(count, generator=generator) for _ in range(epochs)])


def accounting(count, epochs, best_epoch=None):
    if count < 1 or epochs < 1 or (best_epoch is not None and not 1 <= best_epoch <= epochs):
        raise ValueError('invalid consumed/selected sample budget')
    return dict(consumed_assets=count*epochs, unique_assets=count,
                repeated_assets=count*(epochs-1), consumed_slot_decisions=count*epochs*6,
                unique_slot_decisions=count*6, optimizer_updates=epochs*math.ceil(count/128),
                selected_consumed_assets=None if best_epoch is None else count*best_epoch,
                selected_consumed_slot_decisions=None if best_epoch is None else count*best_epoch*6)


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict(model, features):
    with torch.inference_mode():
        return model(features).detach().cpu().numpy()


def save_predictions(output, name, logits, targets, rows, temperatures=None):
    metric, probabilities = summarize(logits, targets, temperatures)
    np.save(output/(name+'-logits.npy'), logits)
    np.save(output/(name+'-probabilities.npy'), probabilities)
    records = []
    for i, row in enumerate(rows):
        decisions = json.loads(json_decision(logits[i], temperatures=temperatures))
        records.append(dict(source_id=row['source_id'], speaker=row['speaker'],
                            speaker_group=row['speaker_group'], condition=row['condition'],
                            targets=row['targets'], probabilities=decisions,
                            predictions={s: max(p, key=p.__getitem__)
                                         for s, p in decisions.items()}))
    (output/(name+'-predictions.json')).write_text(json.dumps(records, indent=2), encoding='utf-8')
    return metric


def fit_arm(raw, targets, rows, provenance, condition, output, *, smoke=False, device='cuda'):
    if provenance['training_only_smoke'] != smoke:
        raise ValueError('smoke/full source scope mismatch')
    if set(raw) != ({'train'} if smoke else {'train', 'calibration', 'validation'}):
        raise ValueError('independent complete validation/calibration required outside smoke')
    if not smoke and [len(rows[s]) for s in ['train', 'calibration', 'validation']] != [
            4080, 398, 889]:
        raise ValueError('fixed full source counts required; no benchmark shrinking')
    protocol = provenance['protocol']
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(status='RUNNING', condition=condition, smoke=smoke,
                  started_utc=datetime.now(UTC).isoformat(), history=[],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                        text=True).strip(),
                  script_sha256=sha(Path(__file__)), python=platform.python_version(),
                  torch=torch.__version__, torch_cuda=torch.version.cuda,
                  numpy=np.__version__, device=device,
                  dtype='float32', seed=17, protocol=protocol, provenance=provenance,
                  validation_evaluated=False, calibration_evaluated=False,
                  encoders_loaded_in_training=False,
                  gpu_peak_scope='readout on cached features; excludes encoder extraction')
    config = {k: v for k, v in report.items() if k != 'history'}
    config_path = output/'effective-config.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    report['effective_config_sha256'] = sha(config_path)
    normalizer = fit_normalizer(*raw['train'])
    np.savez(output/'normalization.npz', **normalizer)
    x = {s: torch.from_numpy(make_inputs(*v, condition, normalizer)).to(device)
         for s, v in raw.items()}
    y = {s: torch.from_numpy(v).to(device) for s, v in targets.items()}
    torch.manual_seed(17)
    model = SlotHead().to(device)
    report['head_parameters'] = sum(p.numel() for p in model.parameters())
    if report['head_parameters'] != protocol['head_parameters']:
        raise ValueError('fixed head capacity differs')
    report['initial_tensor_sha256'] = state_sha(model)
    report['normalization_coefficients'] = sum(v.size for v in normalizer.values())
    report['unique_train_speakers'] = len({r['speaker'] for r in rows['train']})
    report['unique_train_speaker_groups'] = len({r['speaker_group'] for r in rows['train']})
    report['train_mix'] = {c: sum(r['condition'] == c for r in rows['train']) for c in ['p', 'l']}
    encoder_count = {'audio': 8208384, 'video': 114365184, 'joint': 122573568}[condition]
    report['resident_parameters_and_normalization_coefficients'] = (
        encoder_count + report['head_parameters'] + report['normalization_coefficients'])
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        report['gpu'] = torch.cuda.get_device_name()
    epochs = 2 if smoke else protocol['epochs']
    orders = sample_orders(len(rows['train']), epochs)
    np.save(output/'sampling-orders.npy', orders.numpy())
    report['sampling_order_sha256'] = sha(output/'sampling-orders.npy')
    (output/'train-ids.json').write_text(json.dumps([r['source_id'] for r in rows['train']]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                  weight_decay=protocol['weight_decay'])
    best, updates = float('inf'), 0
    step_log = (output/'optimizer-history.jsonl').open('w', encoding='utf-8')
    def publish():
        report['seconds'] = time.perf_counter()-started
        save_progress(output/'progress.json', report)
    publish()
    try:
        for epoch, order in enumerate(orders, 1):
            tick, rolling_ce, norms = time.perf_counter(), 0., []
            model.train()
            for positions in order.split(protocol['batch_size']):
                positions = positions.to(device)
                optimizer.zero_grad()
                slots = model(x['train'][positions]).split(SIZES, dim=-1)
                loss = torch.stack([F.cross_entropy(v, y['train'][positions, j])
                                    for j, v in enumerate(slots)]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(p.grad) for p in model.parameters()]))
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite gradient norm')
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError('nonfinite updated parameter')
                updates += 1
                norms.append(float(norm))
                rolling_ce += float(loss.detach())*len(positions)
                step_log.write(json.dumps(dict(update=updates, epoch=epoch,
                                                samples=len(positions), gradient_norm=float(norm),
                                                learning_rate=protocol['learning_rate'],
                                                seconds=time.perf_counter()-started))+'\n')
            model.eval()
            if not smoke:
                train_metric, _ = summarize(predict(model, x['train']), targets['train'])
                val_logits = predict(model, x['validation'])
                val_metric, _ = summarize(val_logits, targets['validation'])
                report['validation_evaluated'] = True
                report['history'].append(dict(
                    epoch=epoch, rolling_training_ce=rolling_ce/len(rows['train']),
                    train_fixed=train_metric, validation=val_metric,
                    macro_accuracy_gap=train_metric['macro']['accuracy']-val_metric['macro']['accuracy'],
                    macro_nll_gap=val_metric['macro']['nll']-train_metric['macro']['nll'],
                    gradient_norm_mean=float(np.mean(norms)), gradient_norm_max=max(norms),
                    consumed_assets=epoch*len(rows['train']), updates=updates,
                    epoch_seconds=time.perf_counter()-tick))
                if val_metric['macro']['nll'] < best:
                    best = val_metric['macro']['nll']
                    torch.save(model.state_dict(), output/'best.pt')
                print(condition, 'epoch', epoch, 'macro raw validation',
                      val_metric['macro'], flush=True)
            step_log.flush()
            publish()
        torch.save(model.state_dict(), output/'final.pt')
        if smoke:
            selected_name = 'final'
            report['accounting'] = accounting(len(rows['train']), 2)
        else:
            final_logits = predict(model, x['validation'])
            report['final_raw'] = save_predictions(output, 'final-raw', final_logits,
                                                   targets['validation'], rows['validation'])
            report['best_epoch'] = selected_epoch(report['history'])
            report['accounting'] = accounting(len(rows['train']), epochs, report['best_epoch'])
            model.load_state_dict(torch.load(output/'best.pt', map_location=device,
                                              weights_only=True))
            selected_name = 'best'
            cal_logits = predict(model, x['calibration'])
            np.save(output/'calibration-logits.npy', cal_logits)
            temps = select_temperatures(cal_logits, targets['calibration'])
            report['temperatures'] = temps.tolist()
            report['calibration_evaluated'] = True
            logits = predict(model, x['validation'])
            report['best_raw'] = save_predictions(output, 'best-raw', logits,
                                                  targets['validation'], rows['validation'])
            report['best_calibrated'] = save_predictions(output, 'best-calibrated', logits,
                                                         targets['validation'], rows['validation'],
                                                         temps)
            report['final_same_temperature'] = save_predictions(
                output, 'final-same-temperature', final_logits, targets['validation'],
                rows['validation'], temps)
        check_features = x['train'] if smoke else x['validation']
        before_reload = predict(model, check_features)
        reloaded = SlotHead().to(device)
        reloaded.load_state_dict(torch.load(output/(selected_name+'.pt'), map_location=device,
                                            weights_only=True), strict=True)
        reloaded.eval()
        if not np.array_equal(before_reload, predict(reloaded, check_features)):
            raise ValueError('selected checkpoint reload differs on same backend')
        reversed_options = {s: list(reversed(o)) for s, o in OPTIONS.items()}
        if json.loads(json_decision(before_reload[0])) != json.loads(
                json_decision(before_reload[0], reversed_options)):
            raise ValueError('native JSON option permutation differs')
        report.update(status='PASS_TRAIN_ONLY_SMOKE' if smoke else 'PASS_ARM_EXECUTION',
                      reload_bitwise_equal=True, json_option_permutation_exact=True,
                      final_sha256=sha(output/'final.pt'),
                      normalizer_sha256=sha(output/'normalization.npz'),
                      selected_tensor_sha256=state_sha(model), all_gradients_finite=True)
        if not smoke:
            report['best_sha256'] = sha(output/'best.pt')
        if updates != report['accounting']['optimizer_updates']:
            raise ValueError('optimizer/sample accounting differs')
        if device == 'cuda':
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except Exception as error:
        report.update(status='FAIL_INCOMPLETE', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        step_log.close()
        report['ended_utc'] = datetime.now(UTC).isoformat()
        publish()
        (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    raw, targets, rows, provenance = load_data(ROOT, smoke=args.smoke)
    if not torch.cuda.is_available():
        raise ValueError('local CUDA unavailable; no fallback/cloud run is silently selected')
    if torch.cuda.mem_get_info()[0] < 8*1024**3:
        raise ValueError('insufficient idle local GPU memory')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = ROOT/('lombard-grid-pooled-readouts-smoke-01' if args.smoke
                else 'lombard-grid-pooled-readouts-01')
    out.mkdir(exist_ok=False)
    controller = dict(status='RUNNING', smoke=args.smoke, arms={}, cloud_cost=0,
                      pid=os.getpid(), started_utc=datetime.now(UTC).isoformat())
    try:
        for condition in ['audio', 'video', 'joint']:
            controller['arms'][condition] = fit_arm(raw, targets, rows, provenance, condition,
                                                     out/condition, smoke=args.smoke)
            save_progress(out/'progress.json', controller)
        # Rehash/revalidate source and complete feature banks after learning.
        after = load_data(ROOT, smoke=args.smoke)[3]
        if after != provenance:
            raise ValueError('frozen source/feature provenance changed during training')
        arms = list(controller['arms'].values())
        if len({r['initial_tensor_sha256'] for r in arms}) != 1 or len({
                r['sampling_order_sha256'] for r in arms}) != 1:
            raise ValueError('arms used different initialization/sampler orders')
        controller['status'] = 'PASS_TRAIN_ONLY_SMOKE' if args.smoke else 'PASS_THREE_ARM_EXECUTION'
    except Exception as error:
        controller.update(status='FAIL_INCOMPLETE', error=repr(error),
                          traceback=traceback.format_exc())
        raise
    finally:
        controller['ended_utc'] = datetime.now(UTC).isoformat()
        (out/'result.json').write_text(json.dumps(controller, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in controller.items() if k != 'arms'}), flush=True)


if __name__ == '__main__':
    main()
