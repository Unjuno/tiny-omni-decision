"""Unit fault cases must not depend on a real Windows resource backend."""

import subprocess
import sys
from pathlib import Path


def test_pinned_fault_cases_pass_when_real_windows_counter_is_unavailable():
    script = """
from scripts import diagnose_robomnist_lengths as module
def unavailable():
    raise RuntimeError('simulated Windows resource counter unavailable on CI')
module.peak_bytes = unavailable
import pytest
raise SystemExit(pytest.main([
 'tests/test_robomnist_length_probe.py::test_source_or_numeric_failure_saves_incomplete_terminal_and_never_overwrites',
 'tests/test_robomnist_length_probe.py::test_final_readback_resource_growth_cannot_keep_earlier_pass_snapshot',
 '-q',
]))
"""
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf8",
        errors="replace",
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
