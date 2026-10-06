from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tiny_omni_decision.teacher_cache import (
    cache_manifest_sha256,
    canonical_cache_bytes,
    validate_teacher_cache_record,
)


def record(sample_id: str = "s1") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "ordered_options": ["left", "right"],
        "option_labels": ["A", "B"],
        "option_token_ids": [10, 11],
        "teacher_option_logits": [1.5, -0.5],
        "target_index": 0,
        "temperature": 1.0,
        "preprocessing_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "adapter_sha256": "3" * 64,
        "processor_revision": "6befbaca7398925921802abd1f277b495b78b738",
        "sampled_frame_policy": {"num_frames": 8},
        "source_media_sha256": ["4" * 64],
    }


def test_identity_mismatch_fails_closed() -> None:
    actual = record()
    expected = json.loads(json.dumps(actual))
    expected["ordered_options"] = ["right", "left"]
    with pytest.raises(ValueError, match="ordered_options"):
        validate_teacher_cache_record(actual, expected)


def test_manifest_hash_preserves_row_order_and_rejects_duplicates() -> None:
    a, b = record("a"), record("b")
    assert cache_manifest_sha256([a, b]) != cache_manifest_sha256([b, a])
    with pytest.raises(ValueError, match="duplicate"):
        cache_manifest_sha256([a, a])


def test_canonical_bytes_roundtrip() -> None:
    rows = [record("a"), record("b")]
    payload, digest, count = canonical_cache_bytes(rows)
    decoded = [json.loads(line) for line in payload.decode().splitlines()]
    assert decoded == rows
    assert digest == cache_manifest_sha256(rows)
    assert count == 2


def test_builder_cli_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    output = tmp_path / "cache"
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_postquant_teacher_cache.py"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    first = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(output)],
        text=True,
        capture_output=True,
        env=env,
    )
    assert first.returncode == 0, first.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    second = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(output)],
        text=True,
        capture_output=True,
        env=env,
    )
    assert second.returncode == 2
    assert "exists" in second.stderr


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("option_labels", ["B", "A"]),
        ("option_token_ids", [11, 10]),
        ("target_index", 1),
        ("preprocessing_sha256", "a" * 64),
        ("model_sha256", "b" * 64),
        ("adapter_sha256", "c" * 64),
        ("processor_revision", "different-revision"),
        ("sampled_frame_policy", {"num_frames": 4}),
        ("source_media_sha256", ["d" * 64]),
    ],
)
def test_all_cache_identity_fields_fail_closed(field: str, replacement: object) -> None:
    actual = record()
    expected = json.loads(json.dumps(actual))
    expected[field] = replacement
    with pytest.raises(ValueError, match=field):
        validate_teacher_cache_record(actual, expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.update(teacher_option_logits=[float("nan"), 0.0]), "logits"),
        (lambda row: row.update(ordered_options=["left", "left"]), "ordered_options"),
        (lambda row: row.update(option_labels=["A", "A"]), "option_labels"),
        (lambda row: row.update(option_token_ids=[10, 10]), "option_token_ids"),
        (lambda row: row.update(target_index=2), "target_index"),
        (lambda row: row.update(temperature=2.0), "temperature"),
    ],
)
def test_invalid_cache_rows_are_rejected(mutation, match: str) -> None:
    row = record()
    mutation(row)
    with pytest.raises(ValueError, match=match):
        canonical_cache_bytes([row])


def test_builder_rejects_duplicate_json_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_postquant_teacher_cache.py"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    valid = json.dumps(record())
    duplicate = valid.replace(
        '{"sample_id": "s1"',
        '{"sample_id": "s1", "sample_id": "s2"',
        1,
    )
    cases = [
        duplicate + "\n",
        valid.replace("1.5", "NaN", 1) + "\n",
    ]
    for index, payload in enumerate(cases):
        source = tmp_path / f"bad-{index}.jsonl"
        source.write_text(payload, encoding="utf-8")
        output = tmp_path / f"out-{index}"
        result = subprocess.run(
            [sys.executable, str(script), "--input", str(source), "--output", str(output)],
            text=True,
            capture_output=True,
            env=env,
        )
        assert result.returncode == 2
        assert not output.exists()


def test_builder_rejects_protected_input_and_output_paths(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_postquant_teacher_cache.py"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    protected = tmp_path / "sealed-audit"
    protected.mkdir()
    source = protected / "records.jsonl"
    source.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(tmp_path / "out")],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 2
    assert "protected" in result.stderr.lower()

    clean = tmp_path / "records.jsonl"
    clean.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(clean), "--output", str(protected / "cache")],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 2
    assert "protected" in result.stderr.lower()


def test_builder_manifest_records_order_identity(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    rows = [record("b"), record("a")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "cache"
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_postquant_teacher_cache.py"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(output)],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 2
    assert manifest["cache_sha256"] == cache_manifest_sha256(rows)
    assert len(manifest["sample_id_order_sha256"]) == 64
    assert len(manifest["source_jsonl_sha256"]) == 64
