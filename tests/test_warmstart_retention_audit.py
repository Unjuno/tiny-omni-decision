import hashlib
import json
import sys

import pytest

from scripts.audit_warmstart_retention import main


def setup_run(tmp_path, monkeypatch):
    run = tmp_path / "conditional-repair-range-warmstart-01"
    run.mkdir()
    corpus = tmp_path / "conditional-curriculum-01/validation.jsonl"
    corpus.parent.mkdir()
    rows = [dict(id=str(i), group=str(i), task="count", options=["a", "b"], target="a")
            for i in range(2)]
    corpus.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    files = {
        "config.json": {"paths": {str(corpus): hashlib.sha256(corpus.read_bytes()).hexdigest()}},
        "result.json": {"status": "PASS", "steps": 2048, "best_step": 128},
        "history.json": [dict(step=s, selector=1 if s == 128 else 2)
                         for s in range(0, 2049, 128)],
    }
    for step in (0, 128, 2048):
        files[f"predictions-{step}-curriculum.json"] = [
            dict(r, probabilities=[0.9, 0.1] if step else [0.1, 0.9]) for r in rows
        ]
    for name, value in files.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path), "--output", str(output)])
    return run, corpus, output


def test_saved_audit_and_no_overwrite(tmp_path, monkeypatch):
    _, _, output = setup_run(tmp_path, monkeypatch)
    main()
    report = json.loads(output.read_text())
    assert report["comparisons"]["selected"]["step"] == 128
    assert report["comparisons"]["final"]["groups"]["count"]["percentile_95_interval"] == [1, 1]
    with pytest.raises(FileExistsError):
        main()


@pytest.mark.parametrize("failure", ["corpus", "incomplete", "selector", "prediction_order"])
def test_audit_rejects_invalid_inputs(tmp_path, monkeypatch, failure):
    run, corpus, output = setup_run(tmp_path, monkeypatch)
    if failure == "corpus":
        corpus.write_text(corpus.read_text() + "\n", encoding="utf-8")
    elif failure == "prediction_order":
        path = run / "predictions-128-curriculum.json"
        path.write_text(json.dumps(json.loads(path.read_text())[::-1]), encoding="utf-8")
    else:
        path = run / "result.json"
        result = json.loads(path.read_text())
        result["steps" if failure == "incomplete" else "best_step"] = 0
        path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        main()
    assert not output.exists()
