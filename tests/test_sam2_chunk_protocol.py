import pytest

from scripts.probe_sam2_chunked_decoder import FIXED_CHUNKS, FIXED_ORDER
from tiny_omni_decision import sam2_chunked_masks as module


def test_extend_pins_preserves_originals_and_only_adds_new_sources():
    old = {'old-helper.py': 'original'}
    result = module.extend_pinned_sources(old, {'new-runner.py': 'new'})
    assert result == {'old-helper.py': 'original', 'new-runner.py': 'new'}
    assert old == {'old-helper.py': 'original'}


def test_same_existing_pin_can_be_reaffirmed():
    old = {'old-helper.py': 'original'}
    assert module.extend_pinned_sources(old, old) == old


@pytest.mark.parametrize('path', ['sam2_batch_masks.py', 'mask_trajectory_features.py'])
def test_changed_protected_helper_cannot_be_silently_repinned(path):
    old = {path: 'saved-hash'}
    with pytest.raises(ValueError, match='protected source pin'):
        module.extend_pinned_sources(old, {path: 'current-different-hash'})
    assert old == {path: 'saved-hash'}


def test_fixed_complete_matrix_has_same_work_and_reversed_order():
    assert FIXED_CHUNKS == (16, 32, 64)
    assert FIXED_ORDER == (('native', 'chunk16', 'chunk32', 'chunk64'),
                           ('chunk64', 'chunk32', 'chunk16', 'native'))
