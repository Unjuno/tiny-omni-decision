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
