def test_cpu_ci_hook_skips_torch_runner_import_without_optional_dependency(monkeypatch):
    from scripts import pytest_optional_torch_skip as plugin

    class Item:
        name = "test_training_cli_prefers_its_local_src_over_an_installed_other_worktree"

        def __init__(self):
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: None)
    item = Item()
    plugin.pytest_collection_modifyitems(None, [item])

    assert len(item.markers) == 1
    assert item.markers[0].name == "skip"
    assert "PyTorch" in item.markers[0].kwargs["reason"]
