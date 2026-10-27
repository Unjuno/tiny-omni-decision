import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.probe_droid_nested_multimodal import fit_predict


def test_held_labels_do_not_affect_fit():
    rng = np.random.default_rng(17)
    sensor = rng.normal(size=(16, 4)).astype('float32')
    video = rng.normal(size=(16, 3)).astype('float32')
    text = rng.normal(size=(16, 2)).astype('float32')
    labels = np.arange(16) % 7
    fit = np.arange(16) < 12
    held = ~fit
    first = fit_predict(sensor, video, text, labels, fit, held, 'sensor_video_text', (1, 2))
    labels[held] = 0
    second = fit_predict(sensor, video, text, labels, fit, held, 'sensor_video_text', (1, 2))
    assert np.array_equal(first[2], second[2])
    assert np.allclose(first[2].sum(axis=1), 1)
