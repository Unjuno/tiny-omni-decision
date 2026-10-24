import json
from pathlib import Path

import pytest

from scripts.audit_vjepa21_checkpoint import expected_patch_tokens, uniform_frame_indices


def test_uniform_frame_indices_are_even_and_stable():
    assert uniform_frame_indices(9, 5) == (0, 2, 4, 6, 8)
    assert uniform_frame_indices(4, 8) == (0, 0, 1, 1, 2, 2, 3, 3)
    assert uniform_frame_indices(1, 4) == (0, 0, 0, 0)


@pytest.mark.parametrize("frame_count,requested_count", [(0, 1), (1, 0), (-2, 4)])
def test_uniform_frame_indices_reject_invalid_counts(frame_count, requested_count):
    with pytest.raises(ValueError, match="must be positive"):
        uniform_frame_indices(frame_count, requested_count)


def test_vjepa21_image_and_video_patch_token_counts():
    assert expected_patch_tokens(1, 384, 16, 1) == 576
    assert expected_patch_tokens(8, 384, 16, 2) == 2304


def test_patch_token_count_rejects_non_divisible_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        expected_patch_tokens(7, 384, 16, 2)


def test_machine_readable_candidate_inventory_is_explicit_and_conservative():
    path = Path(__file__).parents[1] / "docs/pretrained_reuse/R1_CANDIDATE_INVENTORY.json"
    inventory = json.loads(path.read_text(encoding="utf-8"))
    candidates = inventory["candidates"]
    candidate_ids = [candidate["id"] for candidate in candidates]
    assert len(candidate_ids) == len(set(candidate_ids))
    required_fields = {
        "id",
        "modality",
        "model_revision",
        "parameters",
        "checkpoint_bytes",
        "parameter_tensor_bytes",
        "primary_citations",
        "license_allowed",
        "license_status",
        "passed_load",
        "has_real_weights",
        "runtime_supported",
        "decision",
    }
    for candidate in candidates:
        assert required_fields <= candidate.keys()
        assert candidate["license_status"] in {"ALLOW", "DENY", "REVIEW", "UNKNOWN"}
        assert candidate["license_allowed"] in {True, False, None}
        if candidate["license_status"] in {"REVIEW", "UNKNOWN"}:
            assert candidate["license_allowed"] is None
        if candidate["passed_load"]:
            assert candidate["has_real_weights"] is True
            assert candidate["runtime_supported"] is True

    vjepa = next(candidate for candidate in candidates if candidate["id"] == "vjepa2-1-vitb-384")
    assert vjepa["license_status"] == "REVIEW"
    assert vjepa["checkpoint_sha256"] == (
        "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
    )
    assert vjepa["parameter_tensor_bytes"] == 347332608

    vivit = next(
        candidate
        for candidate in candidates
        if candidate["id"] == "google-vivit-b-16x2-k400"
    )
    assert vivit["license_status"] == "REVIEW"
    assert vivit["license_allowed"] is None
    assert vivit["passed_load"] is False
    assert vivit["parameters"] is None

    pe_core_t16 = next(candidate for candidate in candidates if candidate["id"] == "pe-core-t16")
    assert pe_core_t16["release_status"] == "pretrained and accessible"
    assert pe_core_t16["checkpoint_sha256"] == (
        "868f89f066b7c1ae51add02d2cb7b925fd051076282773f2ec3c71359fbdaf3b"
    )
    assert pe_core_t16["parameters"] == 69546113
    assert pe_core_t16["passed_load"] is True
    assert pe_core_t16["runtime_supported"] is True
    assert pe_core_t16["runtime_smoke_status"] == "PASS_CPU_FORWARD_REVALIDATION"
    assert pe_core_t16["license_status"] == "REVIEW"
    assert pe_core_t16["license_allowed"] is None
    assert pe_core_t16["task_quality_status"].startswith("measured only on synthetic CLEVR-4")

    videomae = next(
        candidate for candidate in candidates if candidate["id"] == "mcg-nju-videomae-base"
    )
    assert videomae["license_status"] == "DENY"
    assert videomae["license_allowed"] is False
    assert videomae["passed_load"] is False

    moment = next(
        candidate for candidate in candidates if candidate["id"] == "moment-1-small-sensor"
    )
    assert moment["model_revision"] == (
        "AutonLab/MOMENT-1-small@411e288267f82cce86296dbe4d6c8bc533cc162f"
    )
    assert moment["passed_load"] is True
    assert moment["task_quality_status"].startswith("unmeasured;")
    assert moment["pretraining_overlap_status"] == "UNKNOWN"
