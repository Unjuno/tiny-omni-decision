from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_path_a_videoprism_clevr4_runtime.py"
SPEC = importlib.util.spec_from_file_location("verify_path_a_videoprism_clevr4_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY_MODULE
SPEC.loader.exec_module(VERIFY_MODULE)


def test_verifier_loads_runtime_from_bundle_path(tmp_path):
    package = tmp_path / "runtime" / "tiny_omni_decision"
    package.mkdir(parents=True)
    runtime = package / "videoprism_runtime.py"
    runtime.write_text(
        "class VideoPrismClevr4Runtime:\n"
        "    marker = 'from-bundle'\n",
        encoding="utf-8",
    )

    runtime_class = VERIFY_MODULE.load_bundled_runtime(tmp_path)

    assert runtime_class.marker == "from-bundle"
    assert runtime_class.__module__ in sys.modules
