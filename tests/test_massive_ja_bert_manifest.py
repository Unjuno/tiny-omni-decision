from __future__ import annotations

import json
from pathlib import Path


def test_massive_ja_bert_manifest_records_disjoint_split_and_selected_result() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "manifests/pretrained_reuse/massive-ja-bert-readout-20261123.json")
        .read_text(encoding="utf-8")
    )

    assert manifest["status"] == "completed_development_validation_only"
    assert manifest["source"]["publisher_test_downloaded_or_opened"] is False
    assert manifest["split"]["cross_split_source_id_overlap"] == 0
    assert manifest["split"]["cross_split_normalized_utterance_overlap"] == 0
    assert manifest["split"]["project_sealed_audit_opened"] is False
    assert manifest["training"]["selected_epoch_by_validation_nll"] == 8
    assert manifest["training"]["checkpoint_reload_verified"] is True
    assert manifest["bert_readout_reload_verification"]["matching_argmax_rows"] == 2033
    assert manifest["bert_readout_reload_verification"]["max_probability_abs_delta"] == 0.0
    assert manifest["validation"]["selection_affected"] is True
    assert manifest["validation"]["accuracy"] < 0.95
    assert len(manifest["epoch_curve"]) == manifest["training"]["epochs"]
    assert manifest["epoch_curve"][-1]["epoch"] == manifest["training"][
        "selected_epoch_by_validation_nll"
    ]
    comparison = manifest["ruri70m_comparison"]
    assert comparison["model"]["total_prequantization_parameters"] < manifest[
        "model"
    ]["total_prequantization_parameters"]
    assert comparison["validation"]["accuracy"] < manifest["validation"]["accuracy"]
    assert comparison["reload_verification"]["max_probability_abs_delta"] == 0.0
    assert manifest["paired_bert_minus_ruri70m_comparison"]["bootstrap"][
        "iterations"
    ] == 10_000
