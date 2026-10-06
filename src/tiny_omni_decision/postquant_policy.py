"""Strict numerical policy v2, separate from launch-manifest v1.

This module never imports torch, reads a corpus, loads a model, or starts training.
The schedule is indexed by completed optimizer updates, not microbatches.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
TrainedArm = Literal['q2_fixed_recovery',
     'q3_recovery_to_respecialization',
     'q4_high_precision_control']

class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

class Coefficients(_StrictModel):
    option_kl: Nonnegative
    cross_entropy: Nonnegative
    brier: Nonnegative

    @model_validator(mode='after')
    def nonzero(self) -> Self:
        if self.option_kl == self.cross_entropy == self.brier == 0:
            raise ValueError('at least one loss coefficient must be positive')
        return self

class LossWeights(Coefficients):
    stage: Literal['recovery', 'respecialization']

class PostQuantConfig(_StrictModel):
    schema_version: StrictInt
    kind: Literal['postquant_policy']
    arm: TrainedArm
    profile: Literal['smoke', 'comparison']
    total_updates: StrictInt
    recovery_updates: StrictInt
    gradient_accumulation_steps: StrictInt
    temperature: Nonnegative
    full_vocab_kl: Nonnegative
    schedule: Literal['fixed', 'recovery_then_linear']
    recovery: Coefficients
    respecialization_end: Coefficients

    @field_validator('schema_version')
    @classmethod
    def require_v2(cls, value: int) -> int:
        if value != 2:
            raise ValueError('numerical policy requires schema_version 2; no legacy fallback')
        return value

    @model_validator(mode='after')
    def validate_contract(self) -> Self:
        total = 8 if self.profile == 'smoke' else 1024
        fixed = self.arm == 'q2_fixed_recovery'
        recovery = 0 if fixed else total // 4
        if (self.total_updates, self.recovery_updates) != (total, recovery):
            raise ValueError(
                f'arm/profile requires total_updates={total}, recovery_updates={recovery}'
            )
        if self.gradient_accumulation_steps != 4:
            raise ValueError('this candidate requires four microbatches per optimizer update')
        if self.temperature != 1.0 or self.full_vocab_kl != 0.0:
            raise ValueError('only option-only temperature 1.0 is implemented')
        if self.schedule != ('fixed' if fixed else 'recovery_then_linear'):
            raise ValueError('schedule conflicts with experiment arm')
        start, end = (self.recovery, self.respecialization_end)
        if fixed and start != end:
            raise ValueError('fixed Recovery requires identical start/end coefficients')
        if (end.option_kl > start.option_kl
            or end.cross_entropy < start.cross_entropy
            or end.brier != start.brier):
            raise ValueError('require nonincreasing KD, nondecreasing CE and constant Brier')
        return self

def loss_weights_for_update(completed_updates: int, config: PostQuantConfig) -> LossWeights:
    """Return coefficients for the next update; never change optimizer/LR state."""
    if type(completed_updates) is not int or not 0 <= completed_updates < config.total_updates:
        raise ValueError('completed_updates must identify a valid next optimizer update')
    start, end = (config.recovery, config.respecialization_end)
    if config.schedule == 'fixed' or completed_updates < config.recovery_updates:
        return LossWeights(stage='recovery', **start.model_dump())
    if completed_updates == config.total_updates - 1:
        return LossWeights(stage='respecialization', **end.model_dump())
    fraction = (
        (completed_updates - config.recovery_updates)
        / (config.total_updates - config.recovery_updates - 1)
    )
    return LossWeights(stage='respecialization',
         option_kl=start.option_kl + fraction * (end.option_kl - start.option_kl),
         cross_entropy=start.cross_entropy + fraction * (end.cross_entropy - start.cross_entropy),
         brier=start.brier)

def policy_identity(config: PostQuantConfig) -> str:
    """Complete numerical-policy identity; not a cross-arm shared-contract hash."""
    data = json.dumps(config.model_dump(), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

class _UniqueSafeLoader(yaml.SafeLoader):

    def construct_mapping(self, node: Any, deep: bool=False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError('policy YAML requires unique string keys')
            result[key] = self.construct_object(value_node, deep=deep)
        return result

def load_postquant_config(path: Path) -> PostQuantConfig:
    """Read a small local YAML/JSON policy. Duplicate keys and aliases fail closed."""
    pattern = re.compile('(^|[-_.])(audit|sealed|heldout)([-_.]|$)', re.I)
    if any((pattern.search(part) for p in (path, path.resolve()) for part in p.parts)):
        raise ValueError('protected audit/sealed policy path')
    if not path.is_file() or path.is_symlink():
        raise ValueError('policy must be a local regular file, not a symlink')
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('numerical policy exceeds 64 KiB')
    try:
        text = raw.decode('utf-8')
        if any((isinstance(event, yaml.AliasEvent) for event in yaml.parse(text))):
            raise ValueError('policy YAML aliases are unsupported')
        data = yaml.load(text, Loader=_UniqueSafeLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f'invalid policy YAML: {exc}') from exc
    return PostQuantConfig.model_validate(data)
