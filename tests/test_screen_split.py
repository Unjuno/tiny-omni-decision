import subprocess
import sys

import pytest

from scripts.prepare_groundcua_pilot import split_apps


def test_app_split_deterministic_and_disjoint():
    apps = ["7-Zip", "Anki", "Audacity"] + [f"app{i}" for i in range(84)]
    a = split_apps(apps)
    assert a == split_apps(list(reversed(apps)))
    assert list(map(len, a.values())) == [60, 10, 17]
    assert len(set(sum(a.values(), []))) == 87
    assert set(apps[:3]).issubset(a["train"])


def test_inventory_change_fails_closed():
    with pytest.raises(ValueError):
        split_apps(["7-Zip"])


def test_split_import_does_not_require_acquisition_dependencies():
    code = '''
import sys
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"PIL", "requests"}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from scripts.prepare_groundcua_pilot import split_apps
assert callable(split_apps)
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
