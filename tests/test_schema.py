from pathlib import Path

import pytest

from tiny_omni_decision.dataset import audit_license, audit_manifest
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import (
    BaseModelManifest,
    DatasetManifest,
    DecisionExample,
    LicenseProvenance,
    MediaRef,
    TextDecision,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def _example(**updates: object) -> DecisionExample:
    values: dict[str, object] = {
        "id": "source:1",
        "modality": "text",
        "state": "state",
        "question": "question",
        "options": ["yes", "no"],
        "target": "yes",
        "source": "source",
        "source_revision": REVISION,
        "source_record_id": "1",
        "split": "train",
        "provenance": {"license": "CC0-1.0", "commercial_use": True},
    }
    values.update(updates)
    return DecisionExample.model_validate(values)


def test_base_model_example_is_valid() -> None:
    data = load_structured_file(ROOT / "manifests" / "base-model.example.yaml")
    manifest = BaseModelManifest.model_validate(data)
    assert manifest.repo_id == "google/gemma-4-E2B-it-qat-q4_0-unquantized"
    assert "audio" in manifest.modalities


def test_dataset_example_is_valid_and_training_allowed() -> None:
    data = load_structured_file(ROOT / "manifests" / "dataset.example.yaml")
    manifest = DatasetManifest.model_validate(data)
    assert manifest.usage == "training"
    assert audit_manifest(manifest)["project_policy"] == "ALLOW"
    restored = DatasetManifest.model_validate_json(manifest.model_dump_json(by_alias=True))
    assert restored == manifest


def test_all_candidate_manifests_have_immutable_revisions() -> None:
    paths = (ROOT / "manifests" / "candidates").glob("*.yaml")
    manifests = [DatasetManifest.model_validate(load_structured_file(path)) for path in paths]
    assert manifests
    assert all(len(item.revision) == 40 for item in manifests)


def test_text_decision_option_count_and_target() -> None:
    TextDecision(state="s", question="q", options=["x", "y"], target="x")
    with pytest.raises(ValueError):
        TextDecision(state="s", question="q", options=["x", "y"], target="z")
    with pytest.raises(ValueError):
        TextDecision(state="s", question="q", options=[str(i) for i in range(63)], target="0")


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_media_ref_and_multimodal_decision_validation(kind: str) -> None:
    media = MediaRef(kind=kind, uri=f"https://example.test/{kind}")
    item = _example(modality=kind, media=[media])
    assert item.media[0].kind == kind
    assert DecisionExample.model_validate_json(item.model_dump_json()) == item


def test_media_ref_requires_safe_reference() -> None:
    with pytest.raises(ValueError):
        MediaRef(kind="image")
    with pytest.raises(ValueError):
        MediaRef(kind="image", path="../escape.png")


def test_decision_example_option_limits_and_target() -> None:
    for count in (2, 20, 60):
        options = [f"option {i}" for i in range(count)]
        assert _example(options=options, target=options[-1]).target == options[-1]
    with pytest.raises(ValueError):
        _example(options=["only"], target="only")
    with pytest.raises(ValueError):
        _example(options=["a", "b"], target="missing")
    with pytest.raises(ValueError):
        _example(options=["a"] * 63, target="a")


def test_license_allow_review_deny_and_unknown_fail_closed() -> None:
    base = {
        "commercial_use": True,
        "derivative_model_training_allowed": True,
        "redistribution_allowed": True,
        "media_redistribution_allowed": None,
        "has_media": False,
        "trust_status": "trusted",
    }
    assert audit_license("MIT", **base)[0] == "ALLOW"
    assert audit_license("UNKNOWN", **base)[0] == "REVIEW"
    assert audit_license("CC-BY-NC-4.0", **base)[0] == "REVIEW"
    assert audit_license("MIT", **{**base, "commercial_use": False})[0] == "DENY"
    assert audit_license("MIT", **{**base, "commercial_use": None})[0] == "REVIEW"


def test_mixed_license_manifest_needs_component_review() -> None:
    manifest = DatasetManifest(
        dataset_id="mixed/data",
        revision=REVISION,
        split="train",
        modalities=["text"],
        upstream_url="https://example.test/data",
        license="per-source",
        usage="training",
        schema={"state": "state", "question": "question", "options": "options", "target": "target"},
        source_components=[
            {
                "component_id": "permissive",
                "license": "MIT",
                "commercial_use": True,
                "derivative_model_training_allowed": True,
                "redistribution_allowed": True,
            },
            {"component_id": "unknown", "license": "UNKNOWN"},
        ],
    )
    audit = audit_manifest(manifest)
    assert audit["project_policy"] == "REVIEW"
    assert audit["source_components"][1]["policy"] == "REVIEW"


def test_unknown_license_provenance_roundtrips() -> None:
    item = _example(provenance=LicenseProvenance(license="UNKNOWN"))
    assert (
        DecisionExample.model_validate_json(item.model_dump_json()).provenance.license == "UNKNOWN"
    )
