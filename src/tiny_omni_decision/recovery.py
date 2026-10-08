"""Configuration and architecture guards for the student Recovery LoRA."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

RECOVERY_DECODER_LINEAR_SUFFIXES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
RECOVERY_DECODER_PREFIX = "language_model.layers."


def select_decoder_recovery_targets(linear_module_names: Iterable[str]) -> tuple[str, ...]:
    """Select loaded decoder linears only; never include modality towers or bridges."""
    names = tuple(linear_module_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("linear module names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("linear module names must be unique")
    targets = tuple(
        sorted(
            name
            for name in names
            if name.startswith(RECOVERY_DECODER_PREFIX)
            and name.rsplit(".", 1)[-1] in RECOVERY_DECODER_LINEAR_SUFFIXES
        )
    )
    if not targets:
        raise ValueError("loaded model exposes no supported decoder linear targets")
    return targets


def validate_recovery_config(config: Mapping[str, Any]) -> None:
    """Fail closed on recovery configs that unfreeze or retarget the frozen base."""
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Recovery config schema")
    base = config.get("base")
    recovery = config.get("recovery")
    training = config.get("training")
    if not isinstance(base, Mapping) or not isinstance(recovery, Mapping):
        raise ValueError("Recovery config must declare base and recovery mappings")
    if base.get("ternary_weights_frozen") is not True:
        raise ValueError("ternary base weights must remain frozen")
    if base.get("source") != "selected_best_qat_checkpoint":
        raise ValueError("Recovery must start from a selected QAT checkpoint")
    if recovery.get("method") != "lora":
        raise ValueError("Recovery method must be LoRA")
    if recovery.get("target_policy") != "language_model_decoder_all_linear":
        raise ValueError("Recovery targets must be restricted to decoder linears")
    if recovery.get("target_modules") not in (None, []):
        raise ValueError("target modules must be resolved from the loaded architecture")
    rank = recovery.get("rank")
    alpha = recovery.get("alpha")
    dropout = recovery.get("dropout")
    if type(rank) is not int or rank < 1:
        raise ValueError("Recovery rank must be a positive integer")
    if type(alpha) is not int or alpha < 1:
        raise ValueError("Recovery alpha must be a positive integer")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("Recovery dropout must be in [0, 1)")
    if not isinstance(training, Mapping):
        raise ValueError("Recovery config must declare training settings")
    if training.get("device") != "cuda" or training.get("dtype") != "bfloat16":
        raise ValueError("this Recovery run is fixed to CUDA BF16")
    if (
        type(training.get("max_sample_repeats")) is not int
        or training.get("max_sample_repeats") != 1
    ):
        raise ValueError("Recovery training must use unique examples only")
    if training.get("early_stopping") is not False:
        raise ValueError("Recovery run must complete its fixed training budget")
