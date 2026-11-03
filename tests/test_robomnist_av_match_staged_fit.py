import json
from pathlib import Path

import pytest
import torch

from scripts import fit_robomnist_av_match_readout as fit_module
from scripts.fit_robomnist_av_match_readout import (
    REPO_ROOT,
    _load_frozen_fit,
    sha256_file,
)
from scripts.fit_robomnist_av_match_train_only import _validate_train_only_coverage
from tiny_omni_decision.robomnist_av_match_readout import (
    BilinearAVMatchScorer,
    SingleModalityMatchScorer,
)

EXPECTED_VALIDATION_GAPS = {
    "Ar1_Vhigh_Ac1_Rp10_UNC085",
    "Ar1_Vhigh_Ac1_Rp22_UNC304",
}


def test_train_only_stage_accepts_only_the_two_frozen_validation_gaps():
    train = {"train-a", "train-b"}
    validation = {"validation-a", *EXPECTED_VALIDATION_GAPS}
    expected = train | validation
    present = expected - EXPECTED_VALIDATION_GAPS

    missing = _validate_train_only_coverage(
        expected_groups=expected, train_groups=train, present_groups=present
    )

    assert missing == EXPECTED_VALIDATION_GAPS


def test_train_only_stage_rejects_any_missing_training_group():
    train = {"train-a", "train-b"}
    validation = {"validation-a", *EXPECTED_VALIDATION_GAPS}

    try:
        _validate_train_only_coverage(
            expected_groups=train | validation,
            train_groups=train,
            present_groups=(train | validation) - {"train-a"},
        )
    except ValueError as error:
        assert "two expected validation gaps" in str(error) or "training groups" in str(error)
    else:
        raise AssertionError("a missing training feature must fail closed")


def test_train_only_stage_rejects_unexpected_missing_validation_group():
    train = {"train-a", "train-b"}
    validation = {"validation-a", "validation-b", *EXPECTED_VALIDATION_GAPS}

    try:
        _validate_train_only_coverage(
            expected_groups=train | validation,
            train_groups=train,
            present_groups=(train | validation) - {"validation-a", "validation-b"},
        )
    except ValueError as error:
        assert "two expected validation gaps" in str(error)
    else:
        raise AssertionError("unexpected cache gaps must fail closed")


def test_train_only_stage_rejects_feature_groups_outside_the_manifest():
    train = {"train-a", "train-b"}
    validation = {"validation-a", *EXPECTED_VALIDATION_GAPS}
    present = (train | validation) - EXPECTED_VALIDATION_GAPS | {"extra"}

    try:
        _validate_train_only_coverage(
            expected_groups=train | validation,
            train_groups=train,
            present_groups=present,
        )
    except ValueError as error:
        assert "outside the frozen manifest" in str(error)
    else:
        raise AssertionError("unmanifested feature groups must fail closed")


def _frozen_fit_fixture(path: Path):
    path.mkdir()
    config_path = path / "config.json"
    config_path.write_text('{"steps":400}\n', encoding="utf-8")
    checkpoint_path = path / "readouts.pt"
    torch.save(
        {
            "joint": BilinearAVMatchScorer().state_dict(),
            "audio_only": SingleModalityMatchScorer(768).state_dict(),
            "video_only": SingleModalityMatchScorer(1536).state_dict(),
        },
        checkpoint_path,
    )
    implementation_paths = {
        "train_only_script": REPO_ROOT / "scripts" / "fit_robomnist_av_match_train_only.py",
        "readout_fit_script": Path(fit_module.__file__).resolve(),
        "readout_module": (
            REPO_ROOT / "src" / "tiny_omni_decision" / "robomnist_av_match_readout.py"
        ),
        "feature_extractor": REPO_ROOT / "scripts" / "extract_robomnist_av_match_feature_cache.py",
    }
    meta = {
        "status": "TRAIN_COMPLETE_VALIDATION_PENDING",
        "manifest_sha256": "manifest",
        "train_feature_hashes_sha256": "train-features",
        "normalizer_hashes": {"train_only": True},
        "implementation_sha256": {
            key: sha256_file(value) for key, value in implementation_paths.items()
        },
        "training": {"steps": 400, "seed": 17, "validation_used": False},
        "validation_features_read": False,
        "config_sha256": sha256_file(config_path),
        "readout_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_frozen_fit_loader_accepts_pinned_checkpoint_and_all_three_readouts(tmp_path):
    fit_path = tmp_path / "fit"
    _frozen_fit_fixture(fit_path)

    loaded_meta, state = _load_frozen_fit(
        fit_path,
        manifest_sha="manifest",
        train_feature_hashes_sha256="train-features",
        normalizer_hashes={"train_only": True},
        steps=400,
    )

    assert loaded_meta["status"] == "TRAIN_COMPLETE_VALIDATION_PENDING"
    assert set(state) == {"joint", "audio_only", "video_only"}


def test_frozen_fit_loader_rejects_changed_training_features(tmp_path):
    fit_path = tmp_path / "fit"
    _frozen_fit_fixture(fit_path)

    with pytest.raises(ValueError, match="does not match"):
        _load_frozen_fit(
            fit_path,
            manifest_sha="manifest",
            train_feature_hashes_sha256="different-train-features",
            normalizer_hashes={"train_only": True},
            steps=400,
        )
