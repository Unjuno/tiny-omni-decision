import subprocess
import sys
from pathlib import Path


def test_nested_clevrer_utility_imports_split_gate_from_this_checkout():
    repository = Path(__file__).resolve().parents[1]
    command = (
        "import scripts.derive_nested_clevrer_sample; "
        "import tiny_omni_decision.textfree_video as module; "
        "print(module.__file__)"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert Path(result.stdout.strip()).resolve() == (
        repository / "src" / "tiny_omni_decision" / "textfree_video.py"
    ).resolve()
