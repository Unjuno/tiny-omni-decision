import pytest


def names():
    stem = "00_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar1_UNC062_"
    return [
        stem + f"Rx{i}_{kind}.{ext}"
        for i in [1, 2, 3]
        for kind, ext in [("cam", "mp4"), ("mic", "wav")]
    ] + [stem + "Rxall_csi.json", stem + "Rxall_pos.json"]


def test_all_sensor_views_are_one_underlying_repetition_not_independent_samples():
    from scripts.audit_robomnist_source import group_members

    groups = group_members(names())
    assert len(groups) == 1
    g = groups[0]
    assert g["group_id"] == "Ar1_Vhigh_Ac0_Rp01_UNC062"
    assert len(g["input_members"]) == 7
    assert g["supervision_only_member"].endswith("pos.json")


def test_macos_resource_forks_are_not_media_or_extra_repetitions():
    from scripts.audit_robomnist_source import group_members

    assert group_members(
        names() + ["__MACOSX/._00_Ar1_Vhigh_Ac0", "__MACOSX/00_Ar1_Vhigh_Ac0/._bad.mp4"]
    ) == group_members(names())


@pytest.mark.parametrize(
    "bad",
    [
        "00_Ar1_Vhigh_Ac0/Rp01_Ac1_Vhigh_Ar1_UNC062_Rx1_cam.mp4",
        "00_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar2_UNC062_Rx1_cam.mp4",
        "01_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar1_UNC062_Rx1_cam.mp4",
        "../00_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar1_UNC062_Rx1_cam.mp4",
    ],
)
def test_folder_file_label_mismatch_and_unsafe_path_rejected(bad):
    from scripts.audit_robomnist_source import group_members

    with pytest.raises(ValueError):
        group_members([bad])


def test_incomplete_or_duplicate_member_group_is_not_admitted():
    from scripts.audit_robomnist_source import group_members

    with pytest.raises(ValueError):
        group_members(names()[:-1])
    with pytest.raises(ValueError):
        group_members(names() + names()[:1])


def test_internally_consistent_other_class_cannot_be_attributed_to_fixed_archive():
    from scripts.audit_robomnist_source import group_members

    other = [
        n.replace("00_Ar1_Vhigh_Ac0", "01_Ar1_Vhigh_Ac1").replace("/Rp01_Ac0", "/Rp01_Ac1")
        for n in names()
    ]
    assert len(group_members(other)) == 1
    with pytest.raises(ValueError):
        group_members(other, expected_folder="00_Ar1_Vhigh_Ac0")
    assert group_members(names(), expected_folder="00_Ar1_Vhigh_Ac0") == group_members(names())
