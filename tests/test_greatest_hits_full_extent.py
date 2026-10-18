from types import SimpleNamespace

import numpy as np
import pytest


def fixture():
    rgb = np.zeros((15, 256, 456, 3), np.uint8)
    for i in range(15):
        rgb[i] = i
    pts = 2.0045 + np.arange(15) * (1001 / 30000)
    return rgb, pts


def test_review_video_preserves_selected_frames_full_extent_and_real_times():
    from scripts.greatest_hits_full_extent import prepare_review_video

    rgb, pts = fixture()
    original = rgb.copy()
    selected = [0, 2, 4, 6, 8, 10, 12, 14]

    def resize(values, size, method):
        np.testing.assert_array_equal(values, rgb[selected])
        assert size == (162, 288) and method == 'bilinear'
        return np.broadcast_to(values[:, 0:1, 0:1], (8, 162, 288, 3)).astype(np.float32)

    tf = SimpleNamespace(image=SimpleNamespace(resize=resize))
    prepared, positions, metadata = prepare_review_video(rgb, pts, 2., 2.5, tf)
    assert prepared.shape == (8, 288, 288, 3) and prepared.dtype == np.float32
    assert np.all(prepared[:, :63] == 0) and np.all(prepared[:, 225:] == 0)
    for i, source_i in enumerate(selected):
        np.testing.assert_array_equal(prepared[i, 63:225], np.float32(source_i) / 255)
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_allclose(positions[::256, 0], (pts[selected] - 2.) / .5, atol=1e-7)
    assert metadata['selected_indices'] == selected
    assert metadata['selected_pts'] == pts[selected].tolist()
    assert metadata['geometry']['source_box'] == [0, 0, 456, 256]
    assert metadata['policy'] == 'full-extent-bilinear-centered-black-pad-288-v1'
    assert metadata['spatial_coordinates'] == 'encoder canvas including padding'


@pytest.mark.parametrize('case', ['dtype', 'shape', 'count', 'outside', 'duration', 'order'])
def test_review_video_refuses_changed_physical_input(case):
    from scripts.greatest_hits_full_extent import prepare_review_video

    rgb, pts = fixture()
    if case == 'dtype':
        rgb = rgb.astype(np.float32)
    elif case == 'shape':
        rgb = rgb[:, :, :-1]
    elif case == 'count':
        rgb = rgb[:-1]
    elif case == 'outside':
        pts[-1] = 2.6
    elif case == 'order':
        pts[1] = pts[0]
    with pytest.raises(ValueError):
        prepare_review_video(rgb, pts, 2., 3. if case == 'duration' else 2.5, None)
