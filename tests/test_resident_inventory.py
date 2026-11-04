import pytest


def test_resident_inventory_shares_objects_but_counts_identical_copies():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.resident_inventory import parameter_inventory

    model = torch.nn.Linear(3, 2)
    copy = torch.nn.Linear(3, 2)
    copy.load_state_dict(model.state_dict())
    report = parameter_inventory(dict(first=model, alias=model, separate=copy))
    assert report["total_parameters"] == 16
    assert report["components"]["alias"]["incremental_parameters"] == 0
    assert report["components"]["separate"]["incremental_parameters"] == 8
    assert report["tensor_bytes"] == 64
