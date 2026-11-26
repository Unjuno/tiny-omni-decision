import numpy as np
import pytest


def csi():
    return [
        dict(
            module_number=i,
            time_stamp_ns=[1000000000, 1033333333],
            RSS=[-40, -41],
            complex_csi=[["1+2j"] * 256, ["3-4j"] * 256],
            action=9,
        )
        for i in [1, 2, 3]
    ]


def test_csi_preserves_all_native_values_and_excludes_extra_target_metadata():
    from scripts.robomnist_input_arrays import parse_csi

    a, m = parse_csi(csi())
    assert len(a) == 12
    assert a["module_1_real"].dtype == np.float64
    assert a["module_1_timestamps_ns"].dtype == np.int64
    assert a["module_1_real"].shape == (2, 256)
    assert a["module_1_imag"][1, 0] == -4
    assert m[0]["excluded_metadata_keys"] == ["action"]
    assert not any("action" in key for key in a)


@pytest.mark.parametrize(
    "problem", ["duplicate", "timestamp_float", "time_backwards", "complex_bad", "nonfinite"]
)
def test_csi_corruption_is_rejected_without_imputation(problem):
    from scripts.robomnist_input_arrays import parse_csi

    v = csi()
    if problem == "duplicate":
        v[1]["module_number"] = 1
    if problem == "timestamp_float":
        v[0]["time_stamp_ns"][0] = 1.0
    if problem == "time_backwards":
        v[0]["time_stamp_ns"].reverse()
    if problem == "complex_bad":
        v[0]["complex_csi"][0][0] = '__import__("os")'
    if problem == "nonfinite":
        v[0]["complex_csi"][0][0] = "nan+1j"
    with pytest.raises(ValueError):
        parse_csi(v)


def test_pcm_keeps_channels_and_exact_native_scaling():
    from scripts.robomnist_input_arrays import pcm_samples

    raw = np.array([-32768, 32767, 0, 16384], dtype="<i2").tobytes()
    a = pcm_samples(raw, 2, 2)
    assert a.shape == (2, 2)
    assert a.dtype == np.float32
    assert a[0, 0] == -1 and a[1, 1] == 0.5
    with pytest.raises(ValueError):
        pcm_samples(raw[:-1], 2, 2)


def test_video_sample_indices_are_based_on_actual_count_not_nominal450():
    from scripts.robomnist_input_arrays import sample_indices

    assert sample_indices(449) == [0, 224, 448]
    assert sample_indices(451) == [0, 225, 450]
    with pytest.raises(ValueError):
        sample_indices(0)


def test_only_fixed_seven_input_members_can_be_fetched_no_position_gt():
    from scripts.audit_robomnist_source import group_members
    from scripts.fetch_robomnist_input_smoke import fixed_plan
    from tests.test_robomnist_source import names

    members = [
        dict(name=n, size=100, compressed_size=50, crc32="abcdef12", compression=8, offset=i)
        for i, n in enumerate(names())
    ]
    groups = group_members(names())
    bank = dict(groups=groups, members=members, source_archive=dict(name="00_Ar1_Vhigh_Ac0.zip"))
    assert len(fixed_plan(bank)["input_members"]) == 7
    groups[0]["input_members"].append(groups[0]["supervision_only_member"])
    with pytest.raises(ValueError):
        fixed_plan(bank)
