from __future__ import annotations

from pathlib import Path

import yaml


def test_expanded_videoprism_probe_config_is_pinned_and_marks_reused_validation():
    path = (
        Path(__file__).parents[1] / "configs/pretrained_reuse/path_a_video_videoprism_expanded.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment_id"] == "path-a-video-videoprism-clevrer-expanded-v0"
    assert config["source"]["dataset_id"] == "MIT-IBM/CLEVRER"
    assert config["source"]["revision"] == "98b842082ba4f7c18b6b9e3f39145871782a65ef"
    assert config["source"]["train_scenes"] == 256
    assert config["source"]["validation_scenes"] == 200
    assert config["source"]["split_group"] == "video_scene"
    assert config["source"]["normalized_question_content_overlap"] == 0
    assert config["source"]["development_validation_previously_used_for_vjepa_probe"] is True
    assert config["source"]["sealed_audit_loaded"] is False
    assert config["encoder"]["revision"] == "be719a406d563b66f0ac969e7c94bab8e997c81a"
    assert config["encoder"]["frame_count"] == 8
    assert config["encoder"]["frozen"] is True
    assert config["readout"]["checkpoint_selection"] == "minimum_validation_nll"
