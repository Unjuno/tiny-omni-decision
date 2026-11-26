import json
import subprocess
import sys

import pytest
from test_condition_relevance_postfit_io import files, sha, write_json  # noqa: F401


def test_preparer_rejects_another_executing_checkout_before_corpus_access(tmp_path, monkeypatch):
    from scripts import prepare_condition_relevance_comparison as target

    monkeypatch.setattr(target, "verify_pair_files", lambda *args: {})
    with pytest.raises(ValueError, match="executing entrypoint"):
        target.prepare({}, {}, tmp_path / "another-repo", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_refuses_unfinished_pair_without_creating_output(files, tmp_path):  # noqa: F811
    from scripts import prepare_condition_relevance_comparison as target

    run = files["metadata"]["relevance"]
    run["status"] = "RUNNING"
    result = files["runs"]["relevance"] / "result.json"
    write_json(result, run)
    ce_result = files["runs"]["ce"] / "result.json"
    files["metadata"]["ce"]["status"] = "RUNNING"
    write_json(ce_result, files["metadata"]["ce"])
    destination = tmp_path / "comparison"
    response = subprocess.run(
        [
            sys.executable,
            "-m",
            target.__name__,
            "--ce-directory",
            str(files["runs"]["ce"]),
            "--ce-sha256",
            sha(ce_result),
            "--relevance-directory",
            str(files["runs"]["relevance"]),
            "--relevance-sha256",
            sha(result),
            "--source-root",
            str(files["source"]),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert response.returncode != 0
    assert "completed main fit required" in response.stderr
    assert not destination.exists()
    assert json.loads(result.read_text())["status"] == "RUNNING"
