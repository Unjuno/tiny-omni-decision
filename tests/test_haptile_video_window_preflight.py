from __future__ import annotations

import hashlib
import json

from scripts.pretrained_reuse import preflight_haptile_video_windows as preflight


def test_preflight_records_label_blind_video_eligibility(tmp_path, monkeypatch):
    media_root = tmp_path / "validation-corpus"
    (media_root / "data" / "valid_task").mkdir(parents=True)
    media = media_root / "data" / "valid_task" / "episode.fo.mcap"
    media.write_bytes(b"fixed media")
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    receipt_path = media_root / "validation-corpus-receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text('{"source_revision":"pinned"}', encoding="utf-8")
    manifest_hash = preflight.sha256_file(manifest_path)
    action_path = tmp_path / "action.json"
    action = {
        "format": "haptile_validation_action_preflight_v1",
        "input_receipt_sha256": preflight.sha256_file(receipt_path),
        "split_manifest_sha256": manifest_hash,
        "samples": [{
            "path": "data/valid_task/episode.fo.mcap", "sha256": digest,
            "query_time_log_seconds": 12.0, "target": "hold",
        }],
    }
    action_path.write_text(json.dumps(action), encoding="utf-8")
    file_row = {
        "path": "data/valid_task/episode.fo.mcap",
        "size": len(media.read_bytes()),
        "sha256": digest,
        "task": "valid_task",
    }
    monkeypatch.setattr(preflight, "validate_development_receipt", lambda *args: [file_row])
    selected = {topic: [11.0 + index / 7 for index in range(8)] for topic in preflight.TOPICS}
    monkeypatch.setattr(preflight, "_audit_media", lambda *_: (selected, []))
    output_path = tmp_path / "eligibility.json"

    result = preflight.preflight_video_windows(
        receipt_path=receipt_path,
        action_path=action_path,
        split_manifest_path=manifest_path,
        split_name="validation",
        output_path=output_path,
    )

    assert result["eligible_examples"] == 1
    assert result["format"] == "haptile_validation_video_window_eligibility_v2"
    sample = result["eligible_samples"][0]
    assert sample["query_time_log_seconds"] == 12.0
    assert sample["selected_frame_times_log_seconds"] == selected
    assert "target" not in sample
    assert result["selection_uses_labels"] is False
    assert result["sealed_audit_opened"] is False


def test_preflight_rejects_action_from_another_frozen_split_before_media_scan(
    tmp_path, monkeypatch
):
    receipt_path = tmp_path / "validation-receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text('{"source_revision":"pinned"}', encoding="utf-8")
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps({
        "format": "haptile_validation_action_preflight_v1",
        "input_receipt_sha256": preflight.sha256_file(receipt_path),
        "split_manifest_sha256": "0" * 64,
        "samples": [],
    }), encoding="utf-8")
    monkeypatch.setattr(
        preflight, "validate_development_receipt", lambda *args: []
    )
    monkeypatch.setattr(
        preflight, "_audit_media",
        lambda *_: (_ for _ in ()).throw(AssertionError("media scan must not run")),
    )
    output_path = tmp_path / "eligibility.json"

    try:
        preflight.preflight_video_windows(
            receipt_path=receipt_path,
            action_path=action_path,
            split_manifest_path=manifest_path,
            split_name="validation",
            output_path=output_path,
        )
    except ValueError as error:
        assert "different frozen split" in str(error)
    else:
        raise AssertionError("action preflight from another split unexpectedly passed")

    assert not output_path.exists()


def test_preflight_refuses_sealed_audit_even_before_io(tmp_path):
    try:
        preflight.preflight_video_windows(
            receipt_path=tmp_path / "missing-receipt.json",
            action_path=tmp_path / "missing-action.json",
            split_manifest_path=tmp_path / "missing-split.json",
            split_name="sealed_audit",
            output_path=tmp_path / "out.json",
        )
    except ValueError as error:
        assert "sealed audit" in str(error)
    else:
        raise AssertionError("sealed audit unexpectedly passed the split gate")
