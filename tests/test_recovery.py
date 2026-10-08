from __future__ import annotations

import pytest

from tiny_omni_decision.recovery import (
    RECOVERY_DECODER_LINEAR_SUFFIXES,
    select_decoder_recovery_targets,
    validate_recovery_config,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "base": {
            "source": "selected_best_qat_checkpoint",
            "ternary_weights_frozen": True,
        },
        "recovery": {
            "method": "lora",
            "target_policy": "language_model_decoder_all_linear",
            "target_modules": [],
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
        "training": {
            "device": "cuda",
            "dtype": "bfloat16",
            "max_sample_repeats": 1,
            "early_stopping": False,
        },
    }


def test_recovery_targets_only_loaded_decoder_projection_and_mlp_linears() -> None:
    target_names = select_decoder_recovery_targets(
        [
            "model.vision_tower.encoder.layers.0.self_attn.q_proj",
            "model.audio_tower.layers.0.self_attn.v_proj",
            "model.multi_modal_projector.linear",
            "language_model.layers.0.self_attn.q_proj",
            "language_model.layers.0.self_attn.k_proj",
            "language_model.layers.0.self_attn.v_proj",
            "language_model.layers.0.self_attn.o_proj",
            "language_model.layers.0.mlp.gate_proj",
            "language_model.layers.0.mlp.up_proj",
            "language_model.layers.0.mlp.down_proj",
            "model.language_model.embed_tokens",
            "lm_head",
        ]
    )

    assert len(target_names) == 7
    assert all(name.startswith("language_model.layers.") for name in target_names)
    assert {name.rsplit(".", 1)[-1] for name in target_names} == RECOVERY_DECODER_LINEAR_SUFFIXES


def test_recovery_target_selection_is_deterministic_and_fails_closed() -> None:
    paths = [
        "language_model.layers.1.self_attn.v_proj",
        "language_model.layers.0.self_attn.q_proj",
    ]
    assert select_decoder_recovery_targets(paths) == select_decoder_recovery_targets(
        reversed(paths)
    )
    with pytest.raises(ValueError, match="no supported decoder"):
        select_decoder_recovery_targets(["vision_tower.layers.0.self_attn.q_proj"])
    with pytest.raises(ValueError, match="unique"):
        select_decoder_recovery_targets([paths[0], paths[0]])


def test_recovery_config_requires_frozen_ternary_base_and_decoder_scope() -> None:
    validate_recovery_config(_config())

    mutable_base = _config()
    mutable_base["base"]["ternary_weights_frozen"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="must remain frozen"):
        validate_recovery_config(mutable_base)

    broad_targets = _config()
    broad_targets["recovery"]["target_policy"] = "all_linear"  # type: ignore[index]
    with pytest.raises(ValueError, match="restricted to decoder"):
        validate_recovery_config(broad_targets)


def test_recovery_config_rejects_repeats_or_early_stopping() -> None:
    repeated = _config()
    repeated["training"]["max_sample_repeats"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="unique examples"):
        validate_recovery_config(repeated)

    early_stop = _config()
    early_stop["training"]["early_stopping"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="complete its fixed"):
        validate_recovery_config(early_stop)
