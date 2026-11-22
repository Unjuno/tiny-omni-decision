import pytest

from scripts.pretrained_reuse.download_haptile_train_only import (
    _canonical_tree_bytes,
    _next_tree_url,
    select_task_files,
)


def _file(path="data/insert_peg/episode_000.fo.mcap"):
    return {
        "type": "file",
        "oid": "0" * 40,
        "path": path,
        "size": 12,
        "lfs": {"oid": "a" * 64, "size": 12},
    }


def test_select_task_files_accepts_only_expected_pinned_mcap_entries():
    result = select_task_files(
        [_file(), {"type": "file", "path": "data/insert_peg/readme.txt"}],
        "insert_peg",
        1,
    )
    assert [row["path"] for row in result] == ["data/insert_peg/episode_000.fo.mcap"]


@pytest.mark.parametrize(
    ("entry", "task", "count"),
    [
        (_file("data/fold_tshirt/episode_000.fo.mcap"), "insert_peg", 1),
        (_file("data/insert_peg/../fold_tshirt/episode_000.fo.mcap"), "insert_peg", 1),
        (_file(), "insert_peg", 2),
    ],
)
def test_select_task_files_rejects_out_of_task_or_incomplete_inventory(entry, task, count):
    with pytest.raises(ValueError):
        select_task_files([entry], task, count)


def test_select_task_files_rejects_missing_lfs_hash():
    entry = _file()
    entry["lfs"] = {"size": 12}
    with pytest.raises(ValueError, match="metadata"):
        select_task_files([entry], "insert_peg", 1)


def test_select_task_files_accepts_small_git_blob_without_lfs_pointer():
    entry = _file()
    entry["oid"] = "b" * 40
    entry.pop("lfs")
    assert select_task_files([entry], "insert_peg", 1) == [entry]


def test_select_task_files_rejects_missing_git_blob_oid_without_lfs():
    entry = _file()
    entry.pop("lfs")
    entry["oid"] = "short"
    with pytest.raises(ValueError, match="metadata"):
        select_task_files([entry], "insert_peg", 1)


def test_tree_pagination_stays_on_pinned_task_endpoint():
    revision = "c8593da98d3facb7a9293842514f0ee4ad3f2318"
    path = f"/api/datasets/Voxel51/HapTile/tree/{revision}/data/move_cable"
    link = f'<https://huggingface.co{path}?cursor=next&limit=50>; rel="next"'
    assert _next_tree_url(link, path).endswith("cursor=next&limit=50")


def test_tree_pagination_rejects_other_source_paths():
    revision = "c8593da98d3facb7a9293842514f0ee4ad3f2318"
    path = f"/api/datasets/Voxel51/HapTile/tree/{revision}/data/move_cable"
    link = (
        '<https://huggingface.co/api/datasets/other/tree/revision/data/task?cursor=x>'
        '; rel="next"'
    )
    with pytest.raises(ValueError, match="escaped"):
        _next_tree_url(link, path)


def test_canonical_tree_metadata_is_independent_of_api_row_order():
    entries = [_file("data/insert_peg/episode_001.fo.mcap"), _file()]
    assert _canonical_tree_bytes("insert_peg", entries) == _canonical_tree_bytes(
        "insert_peg", list(reversed(entries))
    )
