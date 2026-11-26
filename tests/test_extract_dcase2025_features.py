from __future__ import annotations

import os

from scripts.extract_dcase2025_features import worker_environment


def test_video_worker_environment_prepends_frozen_dependency_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    dependency = tmp_path / "videoprism-site-packages"
    dependency.mkdir()
    monkeypatch.setenv("PYTHONPATH", "existing-parent-path")

    env = worker_environment(repo, dependency)

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(dependency.resolve()),
        str((repo / "src").resolve()),
        str(repo.resolve()),
    ]
    assert os.environ["PYTHONPATH"] == "existing-parent-path"
