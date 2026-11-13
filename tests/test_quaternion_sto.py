import numpy as np
import pytest

from tiny_omni_decision.quaternion_sto import parse_quaternion_sto


def example():
    return ("DataRate=50\nDataType=Quaternion\nendheader\ntime\tpelvis_imu\n"
            "0\t1,0,0,0\n0.02\t-1,0,0,0\n")


def test_preserves_names_times_and_quaternion_sign_without_resampling():
    result = parse_quaternion_sto(example())
    assert result["sensors"] == ("pelvis_imu",)
    np.testing.assert_array_equal(result["timestamps_ms"], [0, 20])
    np.testing.assert_array_equal(result["values"][:, 0, 0], [1, -1])
    assert result["values"].shape == (2, 1, 4)


@pytest.mark.parametrize("old,new", [
    ("DataType=Quaternion", "DataType=double"),
    ("0.02", "0"), ("-1,0,0,0", "nan,0,0,0"),
    ("-1,0,0,0", "0,0,0,0"), ("-1,0,0,0", "1,0,0"),
])
def test_malformed_sto_is_rejected(old, new):
    with pytest.raises(ValueError):
        parse_quaternion_sto(example().replace(old, new))
