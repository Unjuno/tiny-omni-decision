import copy

import pytest

from scripts.greatest_hits_tasks import tasks_for_parent


def rows():
    result = []
    for i in range(12):
        p = f'2015-01-01-00-00-{i:02d}'
        result += tasks_for_parent(p,'train',f'{i:064x}',
            [dict(seconds=1.,fields=['wood','hit','static'])],set())[0]
    return result


def test_smoke_plan_deterministic_unique_complete_permutations_and_targets():
    from scripts.greatest_hits_learning import smoke_plan

    data = rows()
    before = copy.deepcopy(data)
    plan = smoke_plan(data)
    assert len(plan) == 8 and all(len(batch) == 4 for batch in plan)
    flat = [x for batch in plan for x in batch]
    assert len({x['id'] for x in flat}) == 32
    assert smoke_plan(data[::-1]) == plan and data == before
    by_id = {r['id']:r for r in data}
    for item in flat:
        row = by_id[item['id']]
        assert sorted(item['order']) == list(range(len(row['options'])))
        assert item['order'][item['target']] == row['target']


@pytest.mark.parametrize('case',['validation','duplicate','too_few','bad_target'])
def test_bad_smoke_plan_rejected(case):
    from scripts.greatest_hits_learning import smoke_plan

    data = rows()
    if case == 'validation':
        data[0]['split'] = 'validation'
    elif case == 'duplicate':
        data.append(copy.deepcopy(data[0]))
    elif case == 'too_few':
        data = data[:31]
    else:
        data[0]['target'] = len(data[0]['options'])
    with pytest.raises(ValueError):
        smoke_plan(data)


def test_accumulation_equals_mean_loss_gradient():
    torch = pytest.importorskip('torch')
    from scripts.greatest_hits_learning import backward_batch

    x = torch.tensor([[1.,2.],[3.,1.],[2.,-1.],[-1.,2.]])
    y = torch.tensor([0,1,0,1])
    a = torch.nn.Linear(2,2)
    b = copy.deepcopy(a)
    result = backward_batch(a,[(x[i],int(y[i])) for i in range(4)],
        lambda model,item:(model(item[0]),item[1]))
    torch.nn.functional.cross_entropy(b(x),y).backward()
    assert len(result) == 4
    for p,q in zip(a.parameters(),b.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)


def test_accumulation_rejects_nonfinite_loss():
    torch = pytest.importorskip('torch')
    from scripts.greatest_hits_learning import backward_batch

    model = torch.nn.Linear(2,2)
    with pytest.raises(ValueError):
        backward_batch(model,[None]*4,
                       lambda m,item:(m(torch.tensor([float('nan'),1.])),0))
