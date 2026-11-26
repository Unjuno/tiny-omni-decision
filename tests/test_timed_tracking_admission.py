import copy

import pytest

np = pytest.importorskip('numpy')

from scripts import probe_timed_mask_tracking as probe  # noqa: E402


def bundle():
    contract = dict(input_plan=dict(media_path='same.mp4', frames=6, split='train'),
                    source_hashes={'same.mp4': 'pin'}, rows=[dict(frame_index=3, count=1)])
    arrays = dict(query_tyx=np.array([[3, 1.5, 2.5]], np.float32),
                  proposal_indices=np.array([9], np.int64), nonempty=np.array([True]),
                  frame_indices=np.array([0, 3, 5], np.int64),
                  on_or_after_observation=np.array([[False, True, True]]))
    return contract, arrays


@pytest.mark.parametrize('key', ['query_tyx', 'proposal_indices', 'nonempty',
                               'frame_indices', 'on_or_after_observation'])
def test_self_consistent_saved_plan_cannot_replace_pinned_query_bundle(key):
    contract, expected = bundle()
    actual = {k: a.copy() for k, a in expected.items()}
    actual[key].flat[0] = not actual[key].flat[0] if actual[key].dtype == bool else 1
    with pytest.raises(ValueError):
        probe.require_bundle_binding(contract, actual, contract, expected)


@pytest.mark.parametrize('key', ['input_plan', 'source_hashes', 'rows'])
def test_saved_plan_cannot_replace_pinned_media_time_or_sources(key):
    contract, arrays = bundle()
    changed = copy.deepcopy(contract)
    changed[key] = {} if isinstance(contract[key], dict) else []
    with pytest.raises(ValueError):
        probe.require_bundle_binding(changed, arrays, contract, arrays)


def test_exact_reconstructed_bundle_admitted_without_mutation():
    contract, arrays = bundle()
    probe.require_bundle_binding(contract, arrays, contract, arrays)
    np.testing.assert_array_equal(arrays['query_tyx'], [[3, 1.5, 2.5]])


@pytest.mark.parametrize('dtype', [np.float64, np.int64])
def test_replay_rejects_dtype_changes_even_when_values_equal(dtype):
    _, arrays = bundle()
    changed = {k: a.copy() for k, a in arrays.items()}
    changed['query_tyx'] = changed['query_tyx'].astype(dtype)
    with pytest.raises(ValueError):
        probe.require_arrays(changed, arrays)
