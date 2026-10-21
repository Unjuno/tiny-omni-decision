import sys
from types import SimpleNamespace

from scripts.audit_token_ram_transport import process_tree_rss


def test_memory_includes_launcher_and_actual_encoder_descendant(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(NoSuchProcess=ProcessLookupError))
    launcher = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=10))
    encoder = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1000))

    def children(*, recursive):
        assert recursive is True
        return [launcher, encoder]

    parent = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200), children=children)
    assert process_tree_rss(parent) == 1210
