"""Shared update/evaluation/checkpoint path for the fixed five-arm comparison."""

import copy
import random

import numpy as np
import torch

from scripts.greatest_hits_initialization import ARMS
from scripts.greatest_hits_matched import MODALITIES, presentation
from scripts.native_fit import summarize
from scripts.native_long_budget import learning_rate
from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
from tiny_omni_decision.native_candidate import NativeCandidateReadout


def create_model(arm, *, candidate_output_norm=False):
    if arm not in ARMS:
        raise ValueError('registered factorial arm required')
    if candidate_output_norm and arm == 'C0':
        raise ValueError('candidate output normalization requires an AV residual arm')
    return NativeCandidateReadout() if arm == 'C0' else ConditionalAVReadout(
        exchange=arm[1] == '1', gated=arm[3] == '1',
        candidate_output_norm=candidate_output_norm)


def metric_rows(rows):
    return [dict(r, source='greatest-hits', modality='audio_video', target_index=r['target'])
            if 'event_id' in r else dict(r) for r in rows]


def probe_rows(order, joint, common):
    all_rows = [*joint, *common]
    ids = {r['id'] for r in all_rows}
    used = {i for batch in order for i in batch}
    if (not used or not used <= ids or len(ids) != len(all_rows)
            or any(r['split'] != 'train' for r in all_rows)):
        raise ValueError('known unique TRAIN identities required for full fixed probe')
    return {k: [r for r in rows if r['id'] in used]
            for k, rows in (('joint', joint), ('common', common))}


class MatchedStore:
    def __init__(self, joint, common):
        self.joint, self.common = joint, common

    def get(self, row, *, order=None, mode='joint'):
        order = list(range(len(row['options']))) if order is None else order
        if (any(type(i) is not int for i in order)
                or sorted(order) != list(range(len(row['options'])))):
            raise ValueError('complete candidate permutation required')
        if 'event_id' in row:
            return self.joint.get(row['id'], order=order, mode=mode)
        if mode != 'joint':
            raise ValueError('missing-stream view only defined for joint corpus')
        return self.common.get(dict(row, queries=[row['queries'][i] for i in order]))


def update(model, batch, store, step, optimizer):
    if (len(batch) != 8 or any('event_id' not in r for r in batch[:4])
            or tuple(r.get('modality') for r in batch[4:]) != MODALITIES
            or any('event_id' in r for r in batch[4:])):
        raise ValueError('four joint then Text/Image/Audio/Video microbatches required')
    presentations = [presentation(r, (step-1)*8+i) for i, r in enumerate(batch)]
    rate = learning_rate(step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, correct = [], []
    for row, p in zip(batch, presentations, strict=True):
        blocks, q = store.get(row, order=p['order'])
        scores = model(blocks, q)
        if scores.shape != (len(row['options']),) or not torch.isfinite(scores).all():
            raise ValueError('finite complete option logits required')
        loss = torch.nn.functional.cross_entropy(scores[None],
            torch.tensor([p['target']], device=scores.device))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training CE')
        (loss/8).backward()
        losses.append(float(loss.detach()))
        correct.append(int(scores.detach().argmax()) == p['target'])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    for group in optimizer.param_groups:
        group['lr'] = rate
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite updated parameter')
    return dict(step=step, sample_ids=[r['id'] for r in batch], presentations=presentations,
                losses=losses, mean_ce=sum(losses)/8, accuracy=sum(correct)/8,
                gradient_norm=float(norm), lr=rate, cumulative_examples=step*8)


def evaluate(model, rows, store, *, mode='joint', guard=lambda: None):
    if not rows:
        raise ValueError('full nonempty evaluation rows required')
    scores = np.full((len(rows), 40), -1e30, dtype=np.float32)
    diagnostic = isinstance(model, ConditionalAVReadout) and 'event_id' in rows[0]
    gates = np.full((len(rows), 40, 2), -1., dtype=np.float32) if diagnostic else None
    model.eval()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            blocks, q = store.get(row, mode=mode)
            if diagnostic:
                values, details = model(blocks, q, diagnostics=True)
                if details['active']:
                    gates[i, :len(values)] = details['gates'].cpu().numpy()
            else:
                values = model(blocks, q)
            if values.shape != (len(row['options']),) or not torch.isfinite(values).all():
                raise ValueError('complete finite evaluation logits required')
            scores[i, :len(values)] = values.cpu().numpy()
            if i % 256 == 0:
                guard()
    return scores, gates, summarize(metric_rows(rows), scores)


def checkpoint(model, optimizer, *, step, best_step, best_score, plan_sha256, config_sha256):
    rng = np.random.get_state()
    return dict(model={k: v.detach().cpu().clone() if isinstance(v, torch.Tensor)
                       else copy.deepcopy(v) for k, v in model.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()), step=step,
        best_step=best_step, best_score=best_score, sampler_position=8*step,
        sampler_sha256=plan_sha256, config_sha256=config_sha256,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate(), numpy_rng=(rng[0], rng[1].tolist(), *rng[2:]))
