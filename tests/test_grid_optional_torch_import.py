import subprocess
import sys


def test_grid_evaluation_modules_import_without_optional_torch():
    code = r'''
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockTorch())
import scripts.evaluate_grid_direct_speaker_pilot
import scripts.smoke_grid_direct_speaker_quartet
assert "torch" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
