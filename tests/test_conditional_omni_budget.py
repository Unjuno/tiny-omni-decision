"""Count all stored tensor elements; never hide buffers or duplicated components."""

import json
import struct

import numpy as np
import pytest

from scripts.audit_conditional_omni_budget import array_inventory, summarize


def test_numpy_checkpoint_headers_count_all_float_and_integer_elements(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez(path, weight=np.zeros((2, 3), np.float32), bias=np.zeros(2, np.float64),
             counter=np.array(0, np.int64))
    result = array_inventory(path)
    assert result["elements"] == 9
    assert result["tensor_bytes"] == 48
    assert result["tensor_count"] == 3
    with pytest.raises(ValueError, match="object"):
        np.savez(path, hidden=np.array([{}], dtype=object))
        array_inventory(path)


def test_complete_upper_bound_includes_heads_buffers_and_calibration():
    result = summarize({"encoder": dict(elements=190_000_000, tensor_bytes=760_000_000),
                        "head": dict(elements=9_999_990, tensor_bytes=39_999_960)},
                       calibration_count=13)
    assert result["stored_tensor_elements_plus_calibration"] == 200_000_003
    assert result["within_200m_upper_bound"] is False
    assert result["headroom_elements"] == -3
    assert result["tensor_bytes_plus_float64_calibration"] == 800_000_064
    assert result["combined_model_loaded"] is False
    assert result["quality_qualified"] is False


def test_safetensors_inventory_includes_embeddings_and_rejects_bad_offsets(tmp_path):
    path = tmp_path / "weights.safetensors"
    header = {"embedding": dict(dtype="F32", shape=[2, 3], data_offsets=[0, 24]),
              "__metadata__": {"purpose": "fixture"}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(24))
    result = array_inventory(path)
    assert result["elements"] == 6
    assert result["tensor_bytes"] == 24
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(20))
    with pytest.raises(ValueError, match="offsets"):
        array_inventory(path)


def test_torch_inventory_counts_buffers_and_rejects_optimizer_containers(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "head.pt"
    torch.save({"weight": torch.zeros(2, 3), "batch_count": torch.tensor(0)}, path)
    result = array_inventory(path)
    assert result["elements"] == 7
    assert result["tensor_bytes"] == 32
    torch.save({"state": {"step": 3}, "param_groups": []}, path)
    with pytest.raises(ValueError, match="flat tensor-only"):
        array_inventory(path)
