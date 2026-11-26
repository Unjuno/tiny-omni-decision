import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from scripts.lombard_readout_core import (  # noqa: E402
    OPTIONS,
    SlotHead,
    fit_normalizer,
    json_decision,
    make_inputs,
    select_temperatures,
    selected_epoch,
    summarize,
    targets_from_rows,
)


def test_fixed_head_counts_every_parameter_and_all_native_options():
    model = SlotHead()
    assert sum(p.numel() for p in model.parameters()) == 77235
    with torch.inference_mode():
        assert model(torch.zeros(2, 1154)).shape == (2, 51)
    assert [len(v) for v in OPTIONS.values()] == [4, 4, 4, 25, 10, 4]
    with pytest.raises(ValueError):
        model(torch.zeros(2, 1153))


def test_train_normalization_and_absent_stream_masks():
    audio = np.stack([np.zeros(384), np.ones(384)]).astype('float32')
    video = np.stack([np.zeros(768), np.full(768, 4)]).astype('float32')
    norm = fit_normalizer(audio, video)
    np.testing.assert_array_equal(norm['audio_mean'], np.full(384, .5))
    for mode, bits in [('audio', [1, 0]), ('video', [0, 1]), ('joint', [1, 1])]:
        values = make_inputs(audio, video, mode, norm)
        assert values.dtype == np.float32 and values.shape == (2, 1154)
        np.testing.assert_array_equal(values[:, -2:], np.tile(bits, (2, 1)))
        if mode == 'audio':
            assert np.count_nonzero(values[:, 384:1152]) == 0
        if mode == 'video':
            assert np.count_nonzero(values[:, :384]) == 0
    # Applying an extreme held-out observation cannot refit train statistics.
    make_inputs(audio * 100, video * 100, 'joint', norm)
    np.testing.assert_array_equal(norm['audio_mean'], np.full(384, .5))


def test_uniform_metrics_keep_all_slot_sizes_and_correct_macro_denominator():
    targets = np.tile([len(v)-1 for v in OPTIONS.values()], (2, 1))
    metric, probs = summarize(np.zeros((2, 51)), targets)
    sizes = np.array([4, 4, 4, 25, 10, 4])
    assert metric['macro']['accuracy'] == 0
    assert metric['macro']['nll'] == pytest.approx(np.log(sizes).mean())
    assert metric['macro']['brier'] == pytest.approx((1-1/sizes).mean())
    assert metric['macro']['ece'] == pytest.approx((1/sizes).mean())
    assert metric['whole_utterance_accuracy'] == metric['minimum_slot_accuracy'] == 0
    assert probs.shape == (2, 51)
    for item in metric['by_slot'].values():
        assert item['count'] == 2


def test_perfect_metrics_and_complete_target_validation():
    values = np.full((2, 51), -20.)
    start = 0
    for options in OPTIONS.values():
        values[:, start] = 20
        start += len(options)
    metric, _ = summarize(values, np.zeros((2, 6), dtype=int))
    assert metric['whole_utterance_accuracy'] == metric['minimum_slot_accuracy'] == 1
    assert metric['macro']['nll'] < 1e-12
    row = {'targets': {s: opts[0] for s, opts in OPTIONS.items()}}
    assert targets_from_rows([row]).tolist() == [[0]*6]
    row['targets']['letter'] = 'w'
    with pytest.raises(ValueError):
        targets_from_rows([row])


def test_json_option_permutation_is_exact_and_alphabet_is_never_shrunk():
    values = np.arange(51, dtype=float)
    plain = json.loads(json_decision(values))
    orders = {s: list(reversed(o)) for s, o in OPTIONS.items()}
    permuted = json.loads(json_decision(values, orders))
    assert plain == permuted
    for slot, probs in permuted.items():
        assert list(probs) == orders[slot]
        assert sum(probs.values()) == pytest.approx(1)
    assert len(plain['letter']) == 25
    orders['letter'] = orders['letter'][:20]
    with pytest.raises(ValueError):
        json_decision(values, orders)


def test_earliest_unrounded_raw_nll_and_independent_calibration():
    def row(epoch, nll):
        return {'epoch': epoch, 'validation': {'macro': {'nll': nll}}}
    assert selected_epoch([row(1, .1), row(2, .1)]) == 1
    assert selected_epoch([row(2, .1), row(1, .1)]) == 1
    assert selected_epoch([row(1, .10000000001), row(2, .1)]) == 2
    temps = select_temperatures(np.zeros((3, 51)), np.zeros((3, 6), dtype=int))
    np.testing.assert_array_equal(temps, np.full(6, .25))  # earliest tie in fixed grid
    with pytest.raises(ValueError):
        selected_epoch([row(1, float('nan'))])


@pytest.mark.parametrize('fault', ['shape', 'nan', 'target_range', 'target_float', 'temperature'])
def test_invalid_metric_inputs_fail_closed(fault):
    values, targets, temps = np.zeros((2, 51)), np.zeros((2, 6), dtype=int), None
    if fault == 'shape':
        values = values[:, :50]
    elif fault == 'nan':
        values[0, 0] = np.nan
    elif fault == 'target_range':
        targets[0, 3] = 25
    elif fault == 'target_float':
        targets = targets.astype(float)
    else:
        temps = np.zeros(6)
    with pytest.raises(ValueError):
        summarize(values, targets, temps)
