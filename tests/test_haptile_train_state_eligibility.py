from __future__ import annotations

import pytest

from scripts.pretrained_reuse.extract_haptile_train_state_text import (
    select_video_eligible_train_files,
)


def _fixture():
    files = [
        {"path": "data/task_a/ep_1.mcap", "sha256": "hash-a", "task": "task_a"},
        {"path": "data/task_a/ep_2.mcap", "sha256": "hash-b", "task": "task_a"},
    ]
    action = {"samples": [
        {**files[0], "target": "+x"}, {**files[1], "target": "-y"},
    ]}
    eligibility = {
        "format": "haptile_train_video_window_eligibility_v2",
        "split": "train",
        "input_receipt_sha256": "receipt-hash",
        "action_preflight_sha256": "action-hash",
        "split_manifest_sha256": "manifest-hash",
        "selection_uses_labels": False,
        "sealed_audit_opened": False,
        "criteria": {
            "context_seconds": 1.0,
            "frames_per_stream": 8,
            "max_frame_time_error_seconds": 0.1,
            "causal_cutoff_inclusive": True,
            "streams": ["/front-camera", "/wrist-camera", "/tactile-left", "/tactile-right"],
            "excluded_only_for": "insufficient distinct causal frames in fixed window",
        },
        "corpus_examples": 2,
        "eligible_examples": 1,
        "excluded_examples": 1,
        "eligible_samples": [files[0]],
        "excluded_samples": [{**files[1], "video_window_failures": []}],
    }
    return files, action, eligibility


def test_applies_label_blind_video_gate_to_state_rows() -> None:
    files, action, eligibility = _fixture()
    selected, excluded = select_video_eligible_train_files(
        files=files, action=action, eligibility=eligibility,
        receipt_sha256="receipt-hash", action_sha256="action-hash",
        split_manifest_sha256="manifest-hash",
    )
    assert selected == [files[0]]
    assert excluded == 1


@pytest.mark.parametrize("field,value", [
    ("action_preflight_sha256", "wrong"),
    ("selection_uses_labels", True),
    ("sealed_audit_opened", True),
])
def test_rejects_ineligible_or_unsealed_provenance(field, value) -> None:
    files, action, eligibility = _fixture()
    eligibility[field] = value
    with pytest.raises(ValueError, match="provenance differs"):
        select_video_eligible_train_files(
            files=files, action=action, eligibility=eligibility,
            receipt_sha256="receipt-hash", action_sha256="action-hash",
            split_manifest_sha256="manifest-hash",
        )


def test_rejects_incomplete_partition() -> None:
    files, action, eligibility = _fixture()
    eligibility["excluded_samples"] = []
    eligibility["excluded_examples"] = 0
    with pytest.raises(ValueError, match="does not partition"):
        select_video_eligible_train_files(
            files=files, action=action, eligibility=eligibility,
            receipt_sha256="receipt-hash", action_sha256="action-hash",
            split_manifest_sha256="manifest-hash",
        )
