from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_structured_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected an object at the root of {file_path}")

    return data
