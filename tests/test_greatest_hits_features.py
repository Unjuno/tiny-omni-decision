from types import SimpleNamespace

import numpy as np
import pytest


def test_actual_time_coordinates_and_spatial_grid():
    from scripts.greatest_hits_features import event_positions

    pts = np.array([2.01, 2.07, 2.14, 2.20, 2.27, 2.34, 2.40, 2.48])
    p = event_positions(pts, 2.0, 2.5)
    assert p['video'].shape == (2048, 3)
    np.testing.assert_allclose(p['video'][::256, 0], (pts - 2.) / .5, atol=1e-7)
    assert not np.array_equal(p['video'][::256, 0], np.linspace(0, 1, 8))
    np.testing.assert_array_equal(p['video'][0, 1:], [0, 0])
    np.testing.assert_array_equal(p['video'][255, 1:], [1, 1])
    np.testing.assert_allclose(p['speech'][:, 0], np.arange(25) * .04)
    np.testing.assert_array_equal(p['event_audio'], [[.5, 0, 0]])
    assert all(v.dtype == np.float32 for v in p.values())


@pytest.mark.parametrize('case', ['short', 'nan', 'duplicate', 'outside', 'duration'])
def test_positions_fail_closed(case):
    from scripts.greatest_hits_features import event_positions

    pts = np.linspace(0.01, .48, 8)
    if case == 'short':
        pts = pts[:7]
    elif case == 'nan':
        pts[0] = np.nan
    elif case == 'duplicate':
        pts[1] = pts[0]
    elif case == 'outside':
        pts[-1] = .5
    with pytest.raises(ValueError):
        event_positions(pts, 0., 1. if case == 'duration' else .5)


def audio_fixture():
    torch = pytest.importorskip('torch')

    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    class Speech(Frozen):
        def forward(self, x):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 1500, 384))

    class Event(Frozen):
        def forward(self, x):
            return torch.zeros(1, 527), torch.ones(1, 384)

    class Mel(Frozen):
        def forward(self, x):
            self.seen = x.clone()
            return torch.zeros(1, 128, 51)

    class Processor:
        sampling_rate, n_samples = 16000, 480000

        def __call__(self, x, sampling_rate, return_tensors):
            assert sampling_rate == 16000 and return_tensors == 'pt'
            self.seen = x.copy()
            return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

    return Speech().eval(), Processor(), Event().eval(), Mel().eval()


def test_prepared_audio_is_forwarded_without_resampling_or_normalization():
    from scripts.greatest_hits_features import prepared_audio_tokens

    speech, processor, event, mel = audio_fixture()
    a = np.linspace(-2., 2., 8000, dtype=np.float32)
    b = np.linspace(3., -3., 16000, dtype=np.float32)
    x, y = prepared_audio_tokens(speech, processor, event, mel, a, b)
    assert x.shape == (25, 384) and y.shape == (1, 384)
    np.testing.assert_array_equal(processor.seen, a)
    np.testing.assert_array_equal(mel.seen.numpy()[0], b)
    assert not x.requires_grad and not y.requires_grad


@pytest.mark.parametrize('case', ['train', 'grad', 'dtype', 'short', 'nan'])
def test_audio_contract_refuses_invalid_inputs_and_unfrozen_models(case):
    from scripts.greatest_hits_features import prepared_audio_tokens

    speech, processor, event, mel = audio_fixture()
    a, b = np.zeros(8000, np.float32), np.zeros(16000, np.float32)
    if case == 'train':
        event.train()
    elif case == 'grad':
        speech.anchor.requires_grad_(True)
    elif case == 'dtype':
        a = a.astype(np.float64)
    elif case == 'short':
        b = b[:-1]
    else:
        b[0] = np.nan
    with pytest.raises(ValueError):
        prepared_audio_tokens(speech, processor, event, mel, a, b)
