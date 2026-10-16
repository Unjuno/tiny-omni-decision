import copy

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def rows():
    return [dict(id=str(i), split='train', options=['a', 'b'], target=i % 2,
                 event_id=str(i), task='material') for i in range(4)] + [
        dict(id=str(i+4), split='train', options=['a', 'b'], target_index=i % 2,
             source='test', modality=m, task='choice', queries=['a', 'b'])
        for i, m in enumerate(('text', 'image', 'audio', 'video'))]


class Store:
    def get(self, row, *, order=None, mode='joint'):
        order = [0, 1] if order is None else order
        return [], torch.tensor([[1., 2.], [3., 1.]])[order]


def test_update_equals_mean_of_eight_permuted_ce_gradients():
    from scripts.greatest_hits_matched import presentation
    from scripts.greatest_hits_matched_runtime import update
    from scripts.native_long_budget import learning_rate

    model = torch.nn.Linear(2, 1, bias=False)
    model.forward = lambda blocks, q: torch.nn.functional.linear(q, model.weight).squeeze(-1)
    reference = model.weight.detach().clone().requires_grad_()
    batch = rows()
    expected = []
    for i, row in enumerate(batch):
        p = presentation(row, i)
        q = Store().get(row, order=p['order'])[1]
        expected.append(torch.nn.functional.cross_entropy(
            (q @ reference.T).T, torch.tensor([p['target']])))
    torch.stack(expected).mean().backward()
    norm = float(reference.grad.norm())
    torch.nn.utils.clip_grad_norm_([reference], 1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.9)
    actual = update(model, batch, Store(), 1, optimizer)
    torch.testing.assert_close(model.weight, reference.detach()-learning_rate(1)*reference.grad)
    assert actual['mean_ce'] == pytest.approx(float(torch.stack(expected).mean().detach()))
    assert actual['gradient_norm'] == pytest.approx(norm)
    assert actual['sample_ids'] == [r['id'] for r in batch]
    assert actual['cumulative_examples'] == 8


@pytest.mark.parametrize('change', ['short', 'wrong_modality', 'validation', 'nan'])
def test_update_rejects_wrong_batch_or_nonfinite(change):
    from scripts.greatest_hits_matched_runtime import update

    batch = rows()
    model = torch.nn.Linear(2, 1)
    model.forward = lambda blocks, q: model.weight[0] * (float('nan') if change == 'nan' else 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    if change == 'short':
        batch.pop()
    if change == 'wrong_modality':
        batch[-1]['modality'] = 'text'
    if change == 'validation':
        batch[0]['split'] = 'validation'
    with pytest.raises(ValueError):
        update(model, batch, Store(), 1, optimizer)
    assert not optimizer.state


def test_common_and_joint_query_permutation_and_no_mutation():
    from scripts.greatest_hits_matched_runtime import MatchedStore

    class Common:
        def get(self, row):
            return [], row['queries']

    class Joint:
        def get(self, ident, *, order=None, mode='joint'):
            return ident, order, mode

    data = rows()
    old = copy.deepcopy(data)
    store = MatchedStore(Joint(), Common())
    assert store.get(data[4], order=[1, 0]) == ([], ['b', 'a'])
    assert store.get(data[0], order=[1, 0], mode='audio') == ('0', [1, 0], 'audio')
    with pytest.raises(ValueError):
        store.get(data[4], order=[0, 0])
    with pytest.raises(ValueError):
        store.get(data[4], mode='audio')
    assert data == old


def test_probe_full_fixed_consumed_union_original_order():
    from scripts.greatest_hits_matched_runtime import probe_rows

    data = rows()
    actual = probe_rows([['5', '0', '0'], ['4', '1']], data[:4], data[4:])
    assert [r['id'] for r in actual['joint']] == ['0', '1']
    assert [r['id'] for r in actual['common']] == ['4', '5']
    with pytest.raises(ValueError):
        probe_rows([['missing']], data[:4], data[4:])


def test_evaluation_keeps_complete_scores_and_metadata():
    from scripts.greatest_hits_matched_runtime import evaluate, metric_rows

    data = rows()[:4]
    original = copy.deepcopy(data)
    model = torch.nn.Linear(2, 1)
    model.forward = lambda blocks, q: q[:, 0]
    scores, gates, summary = evaluate(model, data, Store())
    assert scores.shape == (4, 40) and np.all(scores[:, 2:] == np.float32(-1e30))
    assert gates is None
    assert summary['overall']['accuracy'] == .5
    converted = metric_rows(data)
    assert converted[0]['target_index'] == 0
    assert converted[0]['source'] == 'greatest-hits'
    assert converted[0]['modality'] == 'audio_video'
    assert data == original


def test_checkpoint_preserves_extra_state_optimizer_and_rng(tmp_path):
    from scripts.greatest_hits_matched_runtime import checkpoint, create_model

    model = create_model('E1G1')
    optimizer = torch.optim.AdamW(model.parameters())
    value = checkpoint(model, optimizer, step=0, best_step=0, best_score=1.,
                       plan_sha256='plan', config_sha256='config')
    path = tmp_path/'state.pt'
    torch.save(value, path)
    restored = torch.load(path, weights_only=True)
    create_model('E1G1').load_state_dict(restored['model'], strict=True)
    with pytest.raises(ValueError, match='policy'):
        create_model('E0G0').load_state_dict(restored['model'], strict=True)
    assert restored['optimizer']['state'] == {}
    assert restored['sampler_position'] == 0
    assert restored['numpy_rng'] and restored['python_rng']
    assert torch.equal(restored['torch_rng'], torch.get_rng_state()) is False


def test_real_conditional_gate_capture_and_missing_bypass():
    from scripts.greatest_hits_matched_runtime import create_model, evaluate
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    class RealStore:
        def __init__(self):
            self.blocks = [NativeTokenBlock(k, torch.randn(4, width), torch.randn(4, 3),
                torch.ones(4, dtype=torch.bool)) for k, width in [('speech', 384), ('video', 768)]]
            self.queries = torch.randn(2, 384)

        def get(self, row, *, mode='joint'):
            blocks = self.blocks if mode == 'joint' else self.blocks[:1]
            return blocks, self.queries

    model = create_model('E1G1')
    store = RealStore()
    scores, gates, _ = evaluate(model, rows()[:4], store)
    assert gates.shape == (4, 40, 2)
    assert np.all((gates[:, :2] >= 0) & (gates[:, :2] <= 1))
    assert np.all(gates[:, 2:] == -1)
    missing, gates, _ = evaluate(model, rows()[:4], store, mode='audio')
    assert np.all(gates == -1)
    with torch.no_grad():
        expected = model.baseline(store.blocks[:1], store.queries).numpy()
    np.testing.assert_array_equal(missing[0, :2], expected)
    assert np.isfinite(scores).all()
