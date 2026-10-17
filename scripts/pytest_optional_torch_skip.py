"""Skip only the training-runner source-path check when optional Torch is absent."""

import importlib.util

import pytest

TORCH_REQUIRED_TESTS = {
    "test_training_cli_prefers_its_local_src_over_an_installed_other_worktree"
}


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("torch") is not None:
        return
    skip = pytest.mark.skip(reason="PyTorch is an optional dependency for training-runner tests")
    for item in items:
        if item.name in TORCH_REQUIRED_TESTS:
            item.add_marker(skip)
