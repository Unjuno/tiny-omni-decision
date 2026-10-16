"""Fixed TRAIN-only mechanics plan and correctly normalized accumulation."""

import hashlib


def smoke_plan(rows):
    if len(rows) < 32 or len({r['id'] for r in rows}) != len(rows) or any(
            r['split'] != 'train' or type(r['target']) is not int
            or not 0 <= r['target'] < len(r['options']) for r in rows):
        raise ValueError('unique valid TRAIN rows, at least32, required')
    def digest(value):
        return hashlib.sha256(('17:'+value).encode()).hexdigest()
    plan = []
    for row in sorted(rows,key=lambda r:digest(r['id']))[:32]:
        order = sorted(range(len(row['options'])), key=lambda i:digest(row['id']+':'+str(i)))
        plan.append(dict(id=row['id'],order=order,target=order.index(row['target'])))
    return [plan[i:i+4] for i in range(0,32,4)]


def backward_batch(model, batch, forward):
    if len(batch) != 4:
        raise ValueError('exactly four microbatches per update required')
    import torch
    losses = []
    for item in batch:
        scores, target = forward(model,item)
        if (scores.ndim != 1 or len(scores) < 2 or not torch.isfinite(scores).all()
                or type(target) is not int or not 0 <= target < len(scores)):
            raise ValueError('finite logits and aligned target required')
        loss = torch.nn.functional.cross_entropy(scores[None],
            torch.tensor([target],device=scores.device))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training CE')
        (loss/4).backward()
        losses.append(float(loss.detach()))
    return losses
