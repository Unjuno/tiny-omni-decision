import json
import sys

import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize("invalid", ["missing_arm", "unfinished", "bad_hash"])
def test_cli_persists_failure_before_gpu_for_unadmitted_selection(tmp_path, monkeypatch, invalid):
    from scripts import score_greatest_hits_replacements as cli
    from scripts.native_video_cache import sha

    monkeypatch.setattr(cli, "ROOT", tmp_path)

    # Only Git metadata is stubbed; real receipt serialization and SHA checks run.
    def output(command, **kwargs):
        assert command[0] == "git", "GPU must not be queried for invalid admission"
        return "" if kwargs.get("text") else b""

    monkeypatch.setattr(cli.subprocess, "check_output", output)
    manifest = {}
    for arm in cli.ARMS:
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "result.json").write_text(
            json.dumps(dict(status="RUNNING", arm=arm)), encoding="utf8"
        )
        manifest[arm] = dict(path=str(folder), result_sha256=sha(folder / "result.json"))
    if invalid == "missing_arm":
        manifest.pop("E1G1")
    if invalid == "bad_hash":
        manifest["C0"]["result_sha256"] = "0" * 64
    path = tmp_path / "selections.json"
    path.write_text(json.dumps(manifest), encoding="utf8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--selection-set",
            str(path),
            "--selection-set-sha256",
            sha(path),
            "--audit",
            str(tmp_path / "audit"),
            "--arm",
            "C0",
            "--max-seconds",
            "120",
        ],
    )
    with pytest.raises(ValueError):
        cli.main()
    result_path = tmp_path / "greatest-hits-selected-replacements-C0-01/result.json"
    result = json.loads(result_path.read_text(encoding="utf8"))
    assert result["status"] == "INCOMPLETE"
    assert result["model_forwards"] == result["optimizer_updates"] == 0
    assert result["error"] and result["traceback"]
    original = result_path.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert result_path.read_bytes() == original
