import tomllib
from pathlib import Path


def test_ml_extra_declares_psutil_required_by_jsonl_workers() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    ml_dependencies = project["project"]["optional-dependencies"]["ml"]

    assert "psutil>=5.9,<8" in ml_dependencies
