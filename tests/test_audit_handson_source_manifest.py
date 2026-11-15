import hashlib
import io
import json

import pytest

from scripts.audit_handson_source_manifest import audit_stream


def _line(split, task, session, *, extra=None):
    row = {
        "split": split,
        "source_task": task,
        "collection_session_id": session,
        "prompt": "private/example text not emitted",
        "record_id": "do-not-emit",
    }
    if extra:
        row.update(extra)
    return json.dumps(row, sort_keys=True).encode() + b"\n"


def _audit(payload):
    return audit_stream(
        io.BytesIO(payload),
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_aggregates_task_sessions_without_emitting_record_content():
    payload = b"".join(
        (
            _line("train", "task-a", "session-a"),
            _line("train", "task-a", "session-a"),
            _line("train", "task-b", "session-b"),
            _line("validation", "task-a", "session-c"),
            _line("validation", "task-b", "session-d"),
        )
    )

    result = _audit(payload)

    assert result["train_episodes"] == 3
    assert result["validation_episodes"] == 2
    assert result["train_sessions"] == 2
    assert result["validation_sessions"] == 2
    assert result["cross_split_session_overlap"] == 0
    assert result["per_task_validation_sessions"] == {
        "minimum": 1,
        "median": 1.0,
        "maximum": 1,
    }
    serialized = json.dumps(result)
    assert "private/example text" not in serialized
    assert "do-not-emit" not in serialized


def test_rejects_cross_split_session_overlap():
    payload = _line("train", "task-a", "shared") + _line(
        "validation", "task-b", "shared"
    )

    with pytest.raises(ValueError, match="train/validation session overlap"):
        _audit(payload)


def test_rejects_hash_or_size_mismatch_before_returning_metrics():
    payload = _line("train", "task-a", "session-a")

    with pytest.raises(ValueError, match="manifest size mismatch"):
        audit_stream(io.BytesIO(payload), expected_bytes=len(payload) + 1)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        audit_stream(
            io.BytesIO(payload),
            expected_bytes=len(payload),
            expected_sha256="0" * 64,
        )


def test_rejects_invalid_rows_and_missing_group_fields():
    with pytest.raises(ValueError, match="invalid JSONL row"):
        _audit(b"not-json\n")
    with pytest.raises(ValueError, match="collection_session_id"):
        _audit(_line("train", "task-a", "")[:-1] + b"\n")
