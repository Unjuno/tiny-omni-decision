from types import SimpleNamespace

import pytest

from scripts.fetch_grid_smoke_archives import RESERVE, check_space


def test_space_gate_fails_closed(monkeypatch):
    monkeypatch.setattr('shutil.disk_usage', lambda _path: SimpleNamespace(free=RESERVE + 99))
    with pytest.raises(ValueError, match='insufficient'):
        check_space('.', 100)
    check_space('.', 99)
