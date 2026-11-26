import numpy as np
import pytest

from tiny_omni_decision import grid_role_supervision as module


def test_grid_cells_follow_existing_row_major_feature_order():
    cells = module.grid_cells(2)
    assert cells.dtype == np.float64
    assert cells.tolist() == [
        [0, 0, 0.5, 0.5],
        [0.5, 0, 1, 0.5],
        [0, 0.5, 0.5, 1],
        [0.5, 0.5, 1, 1],
    ]


def test_whole_cell_required_partial_small_object_stays_unknown():
    cells = module.grid_cells(2)
    labels, counts = module.cell_source_labels(
        cells, {8: [0, 0, 0.5, 0.5], 9: [0.6, 0.6, 0.9, 0.9]}
    )
    assert labels.tolist() == [8, -1, -1, -1]
    assert counts == dict(total=4, unique=1, unknown=3, ambiguous=0)


def test_partial_competitor_blocks_unique_label_and_keeps_ambiguous():
    cells = module.grid_cells(2)
    labels, counts = module.cell_source_labels(
        cells, {8: [0, 0, 0.5, 0.5], 9: [0.2, 0.2, 0.4, 0.4]}
    )
    assert labels.tolist() == [-2, -1, -1, -1]
    assert counts["ambiguous"] == 1


def test_touching_boundary_is_not_positive_area_overlap():
    labels, _ = module.cell_source_labels(
        module.grid_cells(2), {8: [0, 0, 0.5, 0.5], 9: [0.5, 0, 1, 0.5]}
    )
    assert labels.tolist() == [8, 9, -1, -1]


def test_no_available_boxes_keeps_every_native_cell():
    labels, counts = module.cell_source_labels(module.grid_cells(2), {})
    assert labels.tolist() == [-1] * 4
    assert counts == dict(total=4, unique=0, unknown=4, ambiguous=0)


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_invalid_grid_size_rejected(bad):
    with pytest.raises(ValueError):
        module.grid_cells(bad)


@pytest.mark.parametrize(
    "boxes",
    [
        {-1: [0, 0, 1, 1]},
        {True: [0, 0, 1, 1]},
        {1: [0, 0, 0, 1]},
        {1: [0, 0, 2, 1]},
        {1: [0, 0, np.nan, 1]},
    ],
)
def test_invalid_source_identity_or_geometry_rejected(boxes):
    with pytest.raises(ValueError):
        module.cell_source_labels(module.grid_cells(2), boxes)


def test_pair_roles_are_supervision_only_and_missing_support_not_fabricated():
    labels = np.array([4, 6, -1, 4, 5, -2], np.int64)
    pairs = np.array([[0, 3], [1, 3], [0, 4]], np.int64)
    counts = module.pair_role_counts(labels, pairs, hidden=6, group=[4, 5], cover=4)
    assert counts == dict(
        total=3,
        same_hidden=0,
        same_group=1,
        different_hidden_group=1,
        different_group_group=1,
        same_cover=1,
    )
    with pytest.raises(ValueError):
        module.pair_role_counts(
            labels, np.array([[0, 2]], np.int64), hidden=6, group=[4, 5], cover=4
        )


def test_labels_do_not_change_grid_or_random_state():
    state = np.random.get_state()
    cells = module.grid_cells(4)
    before = cells.tobytes()
    module.cell_source_labels(cells, {7: [0, 0, 1, 1]})
    assert cells.tobytes() == before
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(state, after, strict=True))
