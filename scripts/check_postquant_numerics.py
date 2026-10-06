"""Inspect the numerical policy; opt in to tiny synthetic CPU checks only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.postquant_check import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
