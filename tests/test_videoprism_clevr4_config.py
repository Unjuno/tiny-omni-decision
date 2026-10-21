from __future__ import annotations

from pathlib import Path

import yaml


def test_full_clevr4_videoprism_probe_config_freezes_sources_and_training() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/pretrained_reuse/path_a_image_videoprism_clevr4_full.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment_id"] == "path-a-image-videoprism-clevr4-full-seed17"
    assert config["seed"] == 17
    assert config["task"]["source_revision"] == (
        "sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73"
    )
    assert config["task"]["training_images"] == 8424
    assert config["task"]["checkpoint_selection_validation_images"] == 512
    assert config["task"]["one_time_reporting_validation_images"] == 1595
    assert config["task"]["sealed_audit_loaded"] is False
    assert config["task"]["options_per_taxonomy"] == {
        "color": 10,
        "texture": 10,
        "count": 10,
        "shape": 10,
    }
    assert config["encoder"]["revision"] == "be719a406d563b66f0ac969e7c94bab8e997c81a"
    assert config["encoder"]["checkpoint_sha256"] == (
        "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
    )
    assert config["encoder"]["frames"] == 8
    assert config["encoder"]["frozen"] is True
    assert config["readout"]["trainable_parameters_expected"] == 147713
    assert config["readout"]["checkpoint_selection"] == (
        "minimum_nll_on_prior_512_validation_images"
    )
