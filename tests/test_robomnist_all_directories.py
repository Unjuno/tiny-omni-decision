import pytest

from tests.test_robomnist_source import names


def infos():
    return [
        dict(name=n, directory=False, size=100, compressed_size=50, crc32="abcdef12")
        for n in names()
    ]


def test_budget_tracks_every_read_and_rejects_negative_or_overrun():
    from scripts.audit_robomnist_all_directories import Budget

    b = Budget(total=100, seconds=5)
    b.charge(60, 1)
    assert b.transferred == 60
    for count, elapsed in [(-1, 1), (41, 1), (1, 6)]:
        with pytest.raises(ValueError):
            b.charge(count, elapsed)


def test_metadata_groups_keep_all_views_and_supervision_only_position():
    from scripts.audit_robomnist_all_directories import inspect_infos

    r = inspect_infos(infos(), "00_Ar1_Vhigh_Ac0.zip")
    assert r["repetition_groups"] == 1
    assert r["real_files"] == 8
    assert len(r["groups"][0]["input_members"]) == 7


def test_cross_archive_missing_member_and_bad_size_rejected():
    from scripts.audit_robomnist_all_directories import inspect_infos

    with pytest.raises(ValueError):
        inspect_infos(infos(), "01_Ar1_Vhigh_Ac1.zip")
    with pytest.raises(ValueError):
        inspect_infos(infos()[:-1], "00_Ar1_Vhigh_Ac0.zip")
    bad = infos()
    bad[0]["size"] = -1
    with pytest.raises(ValueError):
        inspect_infos(bad, "00_Ar1_Vhigh_Ac0.zip")


def test_resource_forks_are_counted_but_not_independent_physical_groups():
    from scripts.audit_robomnist_all_directories import inspect_infos

    r = inspect_infos(
        infos()
        + [
            dict(
                name="__MACOSX/._junk", directory=False, size=1, compressed_size=1, crc32="00000000"
            )
        ],
        "00_Ar1_Vhigh_Ac0.zip",
    )
    assert r["repetition_groups"] == 1
    assert r["macos_resource_forks"] == 1
