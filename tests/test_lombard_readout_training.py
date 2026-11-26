import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from scripts.train_lombard_grid_readouts import (  # noqa: E402
    accounting,
    fit_arm,
    sample_orders,
)


def test_orders_repeat_exactly_across_arms_and_cover_each_asset_once_per_epoch():
    first, second = sample_orders(17, 3), sample_orders(17, 3)
    assert torch.equal(first, second)
    for epoch in first:
        assert sorted(epoch.tolist()) == list(range(17))


def test_total_and_selected_counts_include_repetitions_without_conflation():
    result = accounting(4080, 24, 7)
    assert result['consumed_assets'] == 97920
    assert result['unique_assets'] == 4080 and result['repeated_assets'] == 93840
    assert result['selected_consumed_assets'] == 28560
    assert result['consumed_slot_decisions'] == 587520
    assert result['optimizer_updates'] == 768
    with pytest.raises(ValueError):
        accounting(4080, 24, 25)


def test_smoke_never_evaluates_heldout_or_selects_a_best_checkpoint(tmp_path, monkeypatch):
    from scripts import train_lombard_grid_readouts as runner
    def forbidden(*args, **kwargs):
        raise AssertionError('heldout metrics or calibration invoked during smoke')
    monkeypatch.setattr(runner, 'summarize', forbidden)
    monkeypatch.setattr(runner, 'select_temperatures', forbidden)
    rng = np.random.default_rng(17)
    raw = {'train': (rng.normal(size=(4, 384)).astype('float32'),
                     rng.normal(size=(4, 768)).astype('float32'))}
    targets = {'train': np.zeros((4, 6), dtype=np.int64)}
    rows = {'train': [{'source_id': str(i), 'speaker': 'speaker', 'speaker_group': 'speaker',
                       'condition': 'p'} for i in range(4)]}
    protocol = json.loads((Path(__file__).resolve().parents[1]/
                           'configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json').read_text())
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = fit_arm(raw, targets, rows, {'protocol': protocol, 'training_only_smoke': True},
                         'joint', tmp_path/'smoke', smoke=True, device='cpu')
    finally:
        torch.set_num_threads(old_threads)
    assert result['status'] == 'PASS_TRAIN_ONLY_SMOKE'
    assert result['accounting']['optimizer_updates'] == 2
    assert result['accounting']['consumed_assets'] == 8
    assert result['reload_bitwise_equal'] and result['json_option_permutation_exact']
    assert 'validation' not in result and not (tmp_path/'smoke/best.pt').exists()
