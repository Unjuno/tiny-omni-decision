"""Keep pure fault-injection cases independent of the Windows-only run monitor."""

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def length_diagnostic_fault_test_resources(request, monkeypatch):
    cases = {
        "test_source_or_numeric_failure_saves_incomplete_terminal_and_never_overwrites",
        "test_final_readback_resource_growth_cannot_keep_earlier_pass_snapshot",
    }
    if request.node.path.name != "test_robomnist_length_probe.py" or request.node.name not in cases:
        return
    from scripts import diagnose_robomnist_lengths as module

    # These tests inject a source/numerical fault or a time cap, not an OS fault.
    # The committed experimental producer and its source pins stay unchanged.
    monkeypatch.setattr(module, "peak_bytes", lambda: 100)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=4 * 2**30))
