import pytest

np = pytest.importorskip("numpy")
from scripts.replay_trajectory_track_diagnostic import compare  # noqa: E402


def test_exact_frames_only_and_oracle_query():
    geo = np.zeros((2, 3, 4))
    geo[0, :, :2] = [[.2, .2], [.8, .8], [.6, .6]]
    geo[1, :, :2] = .9
    boxes = {0: np.array([.1, .1, .3, .3]), 2: np.array([.5, .5, .7, .7])}
    result = compare(geo, [0, 1, 2], boxes)
    assert result['query_index'] == 0
    assert len(result['points']) == 1
    assert result['points'][0]['frame'] == 2
    assert result['points'][0]['inside']
    assert compare(geo, [0, 1, 2], {})['status'] == 'NO_FRAME0_BOX'
