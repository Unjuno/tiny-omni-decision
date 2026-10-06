"""Validate or package existing Teacher exports; no inference or training."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tiny_omni_decision.postquant_cache import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
