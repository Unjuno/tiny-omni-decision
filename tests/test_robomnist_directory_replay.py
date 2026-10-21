import pytest

from scripts.audit_robomnist_source import group_members
from tests.test_robomnist_all_directories import infos


def bank():
    a = infos()
    return dict(
        source_archive=dict(name="00_Ar1_Vhigh_Ac0.zip"),
        members=a,
        groups=group_members([i["name"] for i in a]),
    )


def test_replay_rebuilds_physical_groups_and_keeps_position_out_of_inputs():
    from scripts.replay_robomnist_directories import verify_bank

    r = verify_bank(bank())
    assert r["repetition_groups"] == 1
    assert r["real_files"] == 8
    assert r["stream_counts"] == {"cam": 3, "mic": 3, "csi": 1, "pos": 1}


def test_replay_rejects_group_identity_or_position_gt_input_corruption():
    from scripts.replay_robomnist_directories import verify_bank

    a = bank()
    a["groups"][0]["group_id"] = "arbitrary"
    with pytest.raises(ValueError):
        verify_bank(a)
    a = bank()
    a["groups"][0]["input_members"].append(a["groups"][0]["supervision_only_member"])
    with pytest.raises(ValueError):
        verify_bank(a)


def test_pass_label_does_not_admit_partial_or_expired_cohort():
    from scripts.replay_robomnist_directories import require_complete

    a = dict(
        status="PASS_COMPLETE_DIRECTORY_INVENTORY_ONLY",
        completed_archives=1,
        rows=[dict(class_index=0)],
        elapsed_seconds=10,
        reserved_transfer_bytes=100,
    )
    with pytest.raises(ValueError):
        require_complete(a)
    a.update(
        completed_archives=60, rows=[dict(class_index=i) for i in range(60)], elapsed_seconds=901
    )
    with pytest.raises(ValueError):
        require_complete(a)


def test_transfer_replay_binds_actual_cumulative_reservations_and_final_total():
    from scripts.replay_robomnist_directories import verify_transfers

    rows = [
        dict(transferred_bytes=100, reserved_transfer_bytes=103),
        dict(transferred_bytes=100, reserved_transfer_bytes=206),
    ]
    assert verify_transfers(rows, 206) == 200
    for bad, final in [
        (rows, 205),
        ([rows[0], dict(transferred_bytes=100, reserved_transfer_bytes=104)], 104),
        ([rows[0], dict(transferred_bytes=100, reserved_transfer_bytes=90)], 90),
        ([dict(transferred_bytes=524289, reserved_transfer_bytes=524300)], 524300),
    ]:
        with pytest.raises(ValueError):
            verify_transfers(bad, final)
