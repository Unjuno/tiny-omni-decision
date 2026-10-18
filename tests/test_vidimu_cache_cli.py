import subprocess
import sys


def test_sealed_audit_is_not_a_cache_cli_option():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.cache_vidimu_training_sensor", "--split", "sealed_audit"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
