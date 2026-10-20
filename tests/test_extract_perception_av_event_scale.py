import pytest

from scripts.extract_perception_av_event_scale import require_hotgpu_gate


def test_cuda_feature_extraction_requires_hotgpu_reservation_environment():
    with pytest.raises(RuntimeError, match="launched through HotUGPU"):
        require_hotgpu_gate({})


def test_cuda_feature_extraction_accepts_safe_hotgpu_reservation():
    environment = {
        "TINY_OMNI_GPU_GATE_PARENT_PID": "100",
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
        "TINY_OMNI_GPU_GATE_FREE_MIB": "13184",
        "TINY_OMNI_GPU_GATE_TOTAL_MIB": "16384",
        "TINY_OMNI_GPU_GATE_UTILIZATION_PCT": "0",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB": "14000",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB": "8000",
    }
    result = require_hotgpu_gate(environment)
    assert result["TINY_OMNI_GPU_GATE_RESERVED_MIB"] == 3200


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("TINY_OMNI_GPU_GATE_RESERVED_MIB", "0"),
        ("TINY_OMNI_GPU_GATE_PARENT_PID", "0"),
        ("TINY_OMNI_GPU_GATE_UTILIZATION_PCT", "5"),
        ("TINY_OMNI_GPU_GATE_FREE_MIB", "7999"),
    ],
)
def test_cuda_feature_extraction_rejects_unsafe_hotgpu_metadata(key, value):
    environment = {
        "TINY_OMNI_GPU_GATE_PARENT_PID": "100",
        "TINY_OMNI_GPU_GATE_RESERVED_MIB": "3200",
        "TINY_OMNI_GPU_GATE_FREE_MIB": "13184",
        "TINY_OMNI_GPU_GATE_TOTAL_MIB": "16384",
        "TINY_OMNI_GPU_GATE_UTILIZATION_PCT": "0",
        "TINY_OMNI_GPU_GATE_BLOCK_IF_FREE_MIB": "14000",
        "TINY_OMNI_GPU_GATE_MIN_CHILD_FREE_MIB": "8000",
    }
    environment[key] = value
    with pytest.raises(RuntimeError, match="reservation metadata"):
        require_hotgpu_gate(environment)
