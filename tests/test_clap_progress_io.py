import json
from pathlib import Path

import pytest

from scripts.clap_progress_io import save_progress


def test_transient_windows_reader_lock_retries_without_partial_json(tmp_path, monkeypatch):
    target = tmp_path / 'progress.json'
    target.write_text('{"old": true}')
    original = Path.replace
    calls = []

    def locked_then_replace(source, destination):
        calls.append(1)
        if len(calls) < 3:
            assert json.loads(target.read_text()) == {'old': True}
            raise PermissionError('sharing violation')
        return original(source, destination)

    monkeypatch.setattr(Path, 'replace', locked_then_replace)
    save_progress(target, {'status': 'PASS'}, attempts=3, delay=0)
    assert len(calls) == 3
    assert json.loads(target.read_text()) == {'status': 'PASS'}


def test_persistent_access_failure_is_not_silenced(tmp_path, monkeypatch):
    target = tmp_path / 'progress.json'
    target.write_text('{"old": true}')

    def denied(*args):
        raise PermissionError('permanent access denial')

    monkeypatch.setattr(Path, 'replace', denied)
    with pytest.raises(PermissionError):
        save_progress(target, {'status': 'PASS'}, attempts=2, delay=0)
    assert json.loads(target.read_text()) == {'old': True}
    assert json.loads(target.with_suffix('.tmp').read_text()) == {'status': 'PASS'}
