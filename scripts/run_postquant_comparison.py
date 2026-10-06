"""Thin entrypoint; all safety checks live in the package, not this script."""
from __future__ import annotations

import sys
from pathlib import Path

# Also support a source checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.postquant_launch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("comparison"))
